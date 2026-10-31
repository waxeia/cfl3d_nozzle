#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "mesh_geometry.hpp"

#ifdef NOZZLE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Vec4 = std::array<double, 4>;
using Vec5 = std::array<double, 5>;
constexpr double tiny = 1.0e-12;

struct Config {
  int nx = 128, ny = 32, max_iter = 3000, report_every = 10;
  double length = 10.0, throat_half_height = 0.5, area_ratio = 1.5;
  double gamma = 1.4, gas_R = 1.0, p0 = 1.0, T0 = 1.0;
  double outlet_pressure = 0.88, cfl = 0.25, residual_drop = 1.0e-6;
  double mass_tolerance = 1.0e-3, stability_tolerance = 1.0e-4;
  double init_mach = 0.2;
  int stability_window = 200;
  std::string flux = "roe", reconstruction = "first", solver = "explicit", backend = "serial";
  std::string init = "uniform", output = "output", name = "case";
  bool choked = false;
};

static std::map<std::string, std::string> readKv(const std::string &path) {
  std::map<std::string, std::string> kv;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  std::string line;
  while (std::getline(in, line)) {
    auto hash = line.find('#'); if (hash != std::string::npos) line.resize(hash);
    auto eq = line.find('='); if (eq == std::string::npos) continue;
    auto trim = [](std::string s) { const char *ws = " \t\r\n";
      s.erase(0, s.find_first_not_of(ws)); s.erase(s.find_last_not_of(ws)+1); return s; };
    kv[trim(line.substr(0, eq))] = trim(line.substr(eq+1));
  }
  return kv;
}
static Config loadConfig(const std::string &path) {
  Config c; auto kv = readKv(path); auto setI=[&](const char*k,int&v){if(kv.count(k))v=std::stoi(kv[k]);};
  auto setD=[&](const char*k,double&v){if(kv.count(k))v=std::stod(kv[k]);};
  auto setS=[&](const char*k,std::string&v){if(kv.count(k))v=kv[k];};
  setI("nx",c.nx); setI("ny",c.ny); setI("max_iter",c.max_iter); setI("report_every",c.report_every); setI("stability_window",c.stability_window);
  setD("length",c.length); setD("throat_half_height",c.throat_half_height); setD("area_ratio",c.area_ratio);
  setD("gamma",c.gamma); setD("R",c.gas_R); setD("p0",c.p0); setD("T0",c.T0); setD("outlet_pressure",c.outlet_pressure);
  setD("cfl",c.cfl); setD("residual_drop",c.residual_drop); setD("mass_tolerance",c.mass_tolerance); setD("stability_tolerance",c.stability_tolerance); setD("init_mach",c.init_mach);
  setS("flux",c.flux); setS("reconstruction",c.reconstruction); setS("solver",c.solver); setS("backend",c.backend); setS("init",c.init); setS("output",c.output); setS("name",c.name);
  if (kv.count("choked")) c.choked = (kv["choked"]=="true" || kv["choked"]=="1");
  for(double value:{c.length,c.throat_half_height,c.area_ratio,c.gamma,c.gas_R,c.p0,c.T0,c.outlet_pressure,c.cfl,c.residual_drop,c.mass_tolerance,c.stability_tolerance,c.init_mach})if(!std::isfinite(value))throw std::runtime_error("configuration contains a non-finite number");
  if (c.nx < 2 || c.ny < 2 || c.max_iter < 1 || c.report_every < 1 ||
      c.length <= 0.0 || c.throat_half_height <= 0.0 || c.area_ratio <= 0.0 ||
      c.gamma <= 1.0 || c.gas_R <= 0.0 || c.p0 <= 0.0 || c.T0 <= 0.0 ||
      c.outlet_pressure <= 0.0 || c.cfl <= 0.0 || c.residual_drop <= 0.0 ||
      c.mass_tolerance <= 0.0 || c.stability_tolerance <= 0.0 || c.init_mach < 0.0 || c.stability_window < 1)
    throw std::runtime_error("invalid grid, geometry, gas, or convergence setting");
  if (c.flux != "roe" && c.flux != "rusanov") throw std::runtime_error("flux must be roe or rusanov");
  if (c.reconstruction != "first" && c.reconstruction != "muscl") throw std::runtime_error("reconstruction must be first or muscl");
  if (c.solver != "explicit" && c.solver != "implicit") throw std::runtime_error("solver must be explicit or implicit");
  if (c.init != "uniform" && c.init != "quasi1d") throw std::runtime_error("init must be uniform or quasi1d");
  if (c.backend != "serial" && c.backend != "openmp") throw std::runtime_error("backend must be serial or openmp");
#ifndef NOZZLE_OPENMP
  if (c.backend == "openmp") throw std::runtime_error("openmp backend requested but executable was built without OpenMP");
#endif
  return c;
}

struct Prim { double rho=1, u=0, v=0, p=1; };
static Prim primitive(const Vec4 &U, double g) {
  Prim q; q.rho=U[0]; q.u=U[1]/q.rho; q.v=U[2]/q.rho;
  q.p=(g-1.0)*(U[3]-0.5*q.rho*(q.u*q.u+q.v*q.v)); return q;
}
static Vec4 conservative(const Prim &q, double g) {
  return {q.rho,q.rho*q.u,q.rho*q.v,q.p/(g-1.0)+0.5*q.rho*(q.u*q.u+q.v*q.v)};
}
static bool valid(const Vec4 &U, double g) { return std::isfinite(U[0]) && std::isfinite(U[3]) && U[0]>1e-10 && primitive(U,g).p>1e-10; }
static Vec4 add(const Vec4&a,const Vec4&b,double s=1) { Vec4 r; for(int k=0;k<4;k++)r[k]=a[k]+s*b[k]; return r; }

struct Face { int left=-1,right=-1; double nx=0,ny=0,length=0,mx=0,my=0; int bc=0; };
struct Cell { double area=0, xc=0,yc=0; std::array<int,4> face{}; };
struct Mesh {
  int nx=0,ny=0; double L=10,hT=.5,rA=1.5; std::vector<nozzle_geometry::Point2> nodes; std::vector<Cell> cells; std::vector<Face> faces;
  int id(int i,int j) const{return j*nx+i;}
  double h(double x) const { const double z=(x-L/2)/(L/2); return hT*(1+(rA-1)*z*z); }
  nozzle_geometry::Point2 node(int i,int j) const { return nodes[j*(nx+1)+i]; }
  void build(int nxi,int nyi,double l,double ht,double ar) {
    if(nxi<2||nyi<2||l<=0.0||ht<=0.0||ar<=0.0)throw std::runtime_error("invalid mesh dimensions or nozzle geometry");
    nx=nxi;ny=nyi;L=l;hT=ht;rA=ar; nodes.resize((nx+1)*(ny+1));
    for(int j=0;j<=ny;j++){ double eta=-1.0+2.0*j/ny; for(int i=0;i<=nx;i++){double x=L*i/nx;nodes[j*(nx+1)+i]={x,eta*h(x)};}}
    cells.assign(nx*ny,{}); faces.clear(); faces.reserve((nx+1)*ny+nx*(ny+1));
    auto addFace=[&](int a,int b,int bc){ auto A=nodes[a],B=nodes[b]; Face f; f.left=-1; f.right=-1; f.bc=bc; f.mx=.5*(A[0]+B[0]);f.my=.5*(A[1]+B[1]);double dx=B[0]-A[0],dy=B[1]-A[1]; f.length=std::hypot(dx,dy); if(!(f.length>0.0))throw std::runtime_error("zero-length mesh face"); f.nx=dy/f.length; f.ny=-dx/f.length; return f; };
    // vertical edges: bottom-to-top; orient from lower-i cell to higher-i cell.
    for(int i=0;i<=nx;i++) for(int j=0;j<ny;j++) { Face f=addFace(j*(nx+1)+i,(j+1)*(nx+1)+i, i==0?1:(i==nx?2:0));
      if(i>0) f.left=id(i-1,j);
      if(i<nx) f.right=id(i,j);
      if(i==0){f.left=id(0,j);f.right=-1;f.nx=-f.nx;f.ny=-f.ny;}
      if(i==nx) f.left=id(nx-1,j);
      int fi=static_cast<int>(faces.size()); faces.push_back(f);
      if(i>0) cells[id(i-1,j)].face[1]=fi; else cells[id(0,j)].face[0]=fi;
      if(i<nx) cells[id(i,j)].face[0]=fi;
    }
    // Horizontal edges run left-to-right; their initial normal points downward.
    for(int i=0;i<nx;i++) for(int j=0;j<=ny;j++) { Face f=addFace(j*(nx+1)+i,j*(nx+1)+i+1,j==0?3:(j==ny?4:0));
      if(j<ny) f.left=id(i,j);
      if(j>0) f.right=id(i,j-1);
      if(j==0){f.left=id(i,0);f.right=-1;}
      if(j==ny){f.left=id(i,ny-1);f.right=-1;f.nx=-f.nx;f.ny=-f.ny;}
      int fi=static_cast<int>(faces.size()); faces.push_back(f);
      if(j>0) cells[id(i,j-1)].face[3]=fi; else cells[id(i,0)].face[2]=fi;
      if(j<ny) cells[id(i,j)].face[2]=fi; else cells[id(i,ny-1)].face[3]=fi;
    }
    for(int j=0;j<ny;j++)for(int i=0;i<nx;i++){
      const auto a=node(i,j),b=node(i+1,j),d=node(i,j+1),e=node(i+1,j+1);
      const auto geometry=nozzle_geometry::quadrilateral(a,b,e,d);
      if(!(geometry.signed_area>0.0)||!std::isfinite(geometry.signed_area)||
         !std::isfinite(geometry.centroid_x)||!std::isfinite(geometry.centroid_y))
        throw std::runtime_error("non-positive or invalid cell geometry at ("+std::to_string(i)+","+std::to_string(j)+")");
      auto &c=cells[id(i,j)];c.area=geometry.signed_area;c.xc=geometry.centroid_x;c.yc=geometry.centroid_y;
    }
  }
  bool geometryOk(double tol=1e-10) const {
    for(std::size_t n=0;n<cells.size();++n){const auto&c=cells[n];if(!(c.area>0.0)||!std::isfinite(c.area)||!std::isfinite(c.xc)||!std::isfinite(c.yc))return false;double sx=0,sy=0;for(int fi:c.face){if(fi<0||fi>=static_cast<int>(faces.size()))return false;const auto&f=faces[fi];if(!(f.length>0.0)||!std::isfinite(f.length)||std::abs(std::hypot(f.nx,f.ny)-1.0)>tol)return false;double s=0;if(f.left==static_cast<int>(n))s=1;else if(f.right==static_cast<int>(n))s=-1;else return false;sx+=s*f.nx*f.length;sy+=s*f.ny*f.length;}if(std::hypot(sx,sy)>tol)return false;} return true;
  }
};

// Normal Euler flux. The FDS branch follows the selected CFL3D ifds=1 role;
// Rusanov is retained as an explicit debugging baseline.
static Vec4 physicalFlux(const Prim&q,double nx,double ny,double g){double un=q.u*nx+q.v*ny;double E=q.p/(g-1)+.5*q.rho*(q.u*q.u+q.v*q.v);return {q.rho*un,q.rho*q.u*un+q.p*nx,q.rho*q.v*un+q.p*ny,(E+q.p)*un};}
static double wavespeed(const Prim&q,double nx,double ny,double g){return std::abs(q.u*nx+q.v*ny)+std::sqrt(g*q.p/q.rho);}
static Vec4 rusanov(const Vec4&UL,const Vec4&UR,double nx,double ny,double g){Prim l=primitive(UL,g),r=primitive(UR,g);auto fl=physicalFlux(l,nx,ny,g),fr=physicalFlux(r,nx,ny,g);double a=std::max(wavespeed(l,nx,ny,g),wavespeed(r,nx,ny,g));Vec4 out;for(int k=0;k<4;k++)out[k]=.5*(fl[k]+fr[k])-.5*a*(UR[k]-UL[k]);return out;}
static Vec4 roeFlux(const Vec4&UL,const Vec4&UR,double nx,double ny,double g){
  Prim l=primitive(UL,g),r=primitive(UR,g); if(l.p<=0||r.p<=0)return rusanov(UL,UR,nx,ny,g); auto fl=physicalFlux(l,nx,ny,g),fr=physicalFlux(r,nx,ny,g);
  double sl=std::sqrt(l.rho),sr=std::sqrt(r.rho),den=sl+sr,u=(sl*l.u+sr*r.u)/den,v=(sl*l.v+sr*r.v)/den;
  double Hl=(UL[3]+l.p)/l.rho,Hr=(UR[3]+r.p)/r.rho,H=(sl*Hl+sr*Hr)/den,un=u*nx+v*ny,ut=-u*ny+v*nx,q2=u*u+v*v,c=std::sqrt(std::max(1e-12,(g-1)*(H-.5*q2))),rho=sl*sr;
  double dr=r.rho-l.rho,dp=r.p-l.p,dun=(r.u-l.u)*nx+(r.v-l.v)*ny,dut=-(r.u-l.u)*ny+(r.v-l.v)*nx,cc=c*c;
  double a1=(dp-rho*c*dun)/(2*cc),a4=(dp+rho*c*dun)/(2*cc),a2=dr-dp/cc,a3=rho*dut; double lam[4]={un-c,un,un,un+c};
  for(double &x:lam){double d=.1*c; if(std::abs(x)<d)x=.5*(x*x/d+d);} Vec4 diss{};
  std::array<Vec4,4> ev={Vec4{1,u-c*nx,v-c*ny,H-c*un},Vec4{1,u,v,H-.5*q2},Vec4{0,-ny,nx,-ut},Vec4{1,u+c*nx,v+c*ny,H+c*un}}; double aa[4]={a1,a2,a3,a4};for(int m=0;m<4;m++)for(int k=0;k<4;k++)diss[k]+=std::abs(lam[m])*aa[m]*ev[m][k];Vec4 out;for(int k=0;k<4;k++)out[k]=.5*(fl[k]+fr[k])-.5*diss[k];return out;
}

struct BlockThomas5 {
  using B=std::array<double,25>; static double &at(B&a,int i,int j){return a[i*5+j];} static double at(const B&a,int i,int j){return a[i*5+j];}
  static bool smallSolve(B M,Vec5 b,Vec5& x){
    for(int k=0;k<5;k++){
      int p=k; for(int i=k+1;i<5;i++) if(std::abs(at(M,i,k))>std::abs(at(M,p,k))) p=i;
      if(std::abs(at(M,p,k))<1e-14) return false;
      if(p!=k){ for(int j=k;j<5;j++) std::swap(at(M,p,j),at(M,k,j)); std::swap(b[p],b[k]); }
      for(int i=k+1;i<5;i++){ double f=at(M,i,k)/at(M,k,k); for(int j=k+1;j<5;j++) at(M,i,j)-=f*at(M,k,j); b[i]-=f*b[k]; }
    }
    for(int i=4;i>=0;i--){double s=b[i];for(int j=i+1;j<5;j++)s-=at(M,i,j)*x[j];x[i]=s/at(M,i,i);} return true;
  }
  static bool solve(std::vector<B> A,std::vector<B> D,std::vector<B> C,std::vector<Vec5>&rhs){
    int n=(int)D.size(); if(n==0) return true;
    // Block Thomas: eliminate the lower block with a 5x5 LU solve at each line.
    for(int i=1;i<n;i++){
      B X{}; Vec5 e{};
      for(int col=0;col<5;col++){e.fill(0);e[col]=1;Vec5 sol{};if(!smallSolve(D[i-1],e,sol))return false;for(int r=0;r<5;r++)at(X,r,col)=sol[r];}
      B prod{}; for(int r=0;r<5;r++)for(int k=0;k<5;k++)for(int m=0;m<5;m++)for(int col=0;col<5;col++)at(prod,r,col)+=at(A[i],r,k)*at(X,k,m)*at(C[i-1],m,col);
      for(int r=0;r<5;r++)for(int col=0;col<5;col++)at(D[i],r,col)-=at(prod,r,col);
      Vec5 y{}; if(!smallSolve(D[i-1],rhs[i-1],y))return false;
      Vec5 Ay{};for(int r=0;r<5;r++)for(int k=0;k<5;k++)Ay[r]+=at(A[i],r,k)*y[k];for(int r=0;r<5;r++)rhs[i][r]-=Ay[r];
    }
    Vec5 z{};if(!smallSolve(D[n-1],rhs[n-1],z))return false;rhs[n-1]=z;
    for(int i=n-2;i>=0;i--){Vec5 Cz{};for(int r=0;r<5;r++)for(int k=0;k<5;k++)Cz[r]+=at(C[i],r,k)*rhs[i+1][k];Vec5 b=rhs[i];for(int r=0;r<5;r++)b[r]-=Cz[r];if(!smallSolve(D[i],b,z))return false;rhs[i]=z;}
    return true;
  }
};

struct Solver {
  struct ExitSample { double mass=0, rho=0, u=0, v=0, p=0; };
  Config c; Mesh mesh; std::vector<Vec4> U, R, face_flux; std::vector<double> dt; std::vector<std::array<double,8>> slopes; std::ofstream hist;
  Solver(const Config&cc):c(cc){mesh.build(c.nx,c.ny,c.length,c.throat_half_height,c.area_ratio);U.assign(mesh.cells.size(),{});R=U;face_flux.assign(mesh.faces.size(),{});dt.resize(U.size());slopes.assign(U.size(),{});}
  bool parallel() const { return c.backend == "openmp"; }
  static double areaMach(double M,double g){
    return (1.0/M)*std::pow(2.0/(g+1.0)*(1.0+0.5*(g-1.0)*M*M),(g+1.0)/(2.0*(g-1.0)));
  }
  static double branchMach(double ratio,double g,bool supersonic){
    if (std::abs(ratio-1.0)<1e-10) return 1.0;
    double lo=supersonic?1.0+1e-10:1e-8, hi=supersonic?20.0:1.0-1e-10;
    for(int n=0;n<100;n++){
      double mid=.5*(lo+hi), v=areaMach(mid,g);
      if (supersonic ? (v<ratio) : (v>ratio)) lo=mid; else hi=mid;
    }
    return .5*(lo+hi);
  }
  void initialize(){
    if(c.init=="quasi1d"){
      double Astar=2.0*c.throat_half_height;
      if(!c.choked){
        double pb=c.outlet_pressure;
        double Me=std::sqrt(2.0/(c.gamma-1.0)*(std::pow(c.p0/pb,(c.gamma-1.0)/c.gamma)-1.0));
        Astar=(2.0*c.throat_half_height*c.area_ratio)/areaMach(Me,c.gamma);
      }
      for(int j=0;j<mesh.ny;j++) for(int i=0;i<mesh.nx;i++){
        int n=mesh.id(i,j); double x=mesh.cells[n].xc;
        double z=(x-c.length*.5)/(c.length*.5);
        double A=2.0*c.throat_half_height*(1.0+(c.area_ratio-1.0)*z*z);
        double M=branchMach(A/Astar,c.gamma,c.choked && x>c.length*.5);
        double T=c.T0/(1.0+.5*(c.gamma-1.0)*M*M);
        double p=c.p0*std::pow(T/c.T0,c.gamma/(c.gamma-1.0));
        double rho=p/(c.gas_R*T), u=M*std::sqrt(c.gamma*c.gas_R*T);
        U[n]=conservative(Prim{rho,u,0,p},c.gamma);
      }
      return;
    }
    Prim q{c.p0/(c.gas_R*c.T0),c.init_mach*std::sqrt(c.gamma*c.gas_R*c.T0),0,c.p0};
    for(auto&u:U)u=conservative(q,c.gamma);
  }
  Vec4 ghost(const Face&f,const Vec4&inside) const {Prim q=primitive(inside,c.gamma);if(f.bc==3||f.bc==4){double un=q.u*f.nx+q.v*f.ny;q.u-=2*un*f.nx;q.v-=2*un*f.ny;return conservative(q,c.gamma);}if(f.bc==1){
      // The outgoing acoustic invariant comes from the interior; total pressure
      // and temperature determine the incoming characteristics.
      double ci=std::sqrt(c.gamma*q.p/q.rho),Jm=q.u-2*ci/(c.gamma-1),a0sq=c.gamma*c.gas_R*c.T0;
      auto fun=[&](double u){double a=std::sqrt(std::max(tiny,a0sq-.5*(c.gamma-1)*u*u));return u-2*a/(c.gamma-1)-Jm;};
      double lo=0,hi=std::sqrt(2*a0sq/(c.gamma+1));for(int n=0;n<80;n++){double mid=.5*(lo+hi);if(fun(mid)>0)hi=mid;else lo=mid;}double u=.5*(lo+hi),T=c.T0-u*u*(c.gamma-1)/(2*c.gamma*c.gas_R),p=c.p0*std::pow(T/c.T0,c.gamma/(c.gamma-1)),rho=p/(c.gas_R*T);return conservative(Prim{rho,u,0,p},c.gamma);
    }if(f.bc==2){double ci=std::sqrt(c.gamma*q.p/q.rho);if(q.u>ci)return inside;double entropy=q.p/std::pow(q.rho,c.gamma),rho=std::pow(c.outlet_pressure/entropy,1/c.gamma),co=std::sqrt(c.gamma*c.outlet_pressure/rho),u=q.u+2*(ci-co)/(c.gamma-1);return conservative(Prim{rho,u,q.v,c.outlet_pressure},c.gamma);}return inside;}
  static double minmod(double a,double b){return a*b<=0?0:(std::abs(a)<std::abs(b)?a:b);}
  void computeSlopes(){
    const int nc=static_cast<int>(slopes.size());
#ifdef NOZZLE_OPENMP
#pragma omp parallel for if(parallel()) schedule(static)
#endif
    for(int n=0;n<nc;n++) slopes[n].fill(0);
    if(c.reconstruction!="muscl")return;
#ifdef NOZZLE_OPENMP
#pragma omp parallel for collapse(2) if(parallel()) schedule(static)
#endif
    for(int j=0;j<mesh.ny;j++)for(int i=0;i<mesh.nx;i++){int n=mesh.id(i,j);for(int k=0;k<4;k++){if(i>0&&i+1<mesh.nx){int l=mesh.id(i-1,j),r=mesh.id(i+1,j);slopes[n][2*k]=minmod((U[n][k]-U[l][k])/(mesh.cells[n].xc-mesh.cells[l].xc),(U[r][k]-U[n][k])/(mesh.cells[r].xc-mesh.cells[n].xc));}if(j>0&&j+1<mesh.ny){int b=mesh.id(i,j-1),t=mesh.id(i,j+1);slopes[n][2*k+1]=minmod((U[n][k]-U[b][k])/(mesh.cells[n].yc-mesh.cells[b].yc),(U[t][k]-U[n][k])/(mesh.cells[t].yc-mesh.cells[n].yc));}}}
  }
  Vec4 faceState(int cell,const Face&f) const {Vec4 q=U[cell];if(c.reconstruction!="muscl")return q;double dx=f.mx-mesh.cells[cell].xc,dy=f.my-mesh.cells[cell].yc;for(int k=0;k<4;k++)q[k]+=0.5*(slopes[cell][2*k]*dx+slopes[cell][2*k+1]*dy);return valid(q,c.gamma)?q:U[cell];}
  void residual(){
    computeSlopes();
    const int nf=static_cast<int>(mesh.faces.size()), nc=static_cast<int>(mesh.cells.size());
#ifdef NOZZLE_OPENMP
#pragma omp parallel for if(parallel()) schedule(static)
#endif
    for(int fi=0;fi<nf;fi++){
      const Face &f=mesh.faces[fi]; Vec4 L=faceState(f.left,f); Vec4 RR=f.right>=0?faceState(f.right,f):ghost(f,L);
      face_flux[fi]=(c.flux=="rusanov")?rusanov(L,RR,f.nx,f.ny,c.gamma):roeFlux(L,RR,f.nx,f.ny,c.gamma);
    }
#ifdef NOZZLE_OPENMP
#pragma omp parallel for if(parallel()) schedule(static)
#endif
    for(int n=0;n<nc;n++){
      Vec4 sum{};
      for(int fi:mesh.cells[n].face){const Face &f=mesh.faces[fi];double sign=(f.left==n)?1.0:-1.0;for(int k=0;k<4;k++)sum[k]+=sign*face_flux[fi][k]*f.length;}
      R[n]=sum;
    }
  }
  bool directionalCorrection(std::vector<Vec4>& delta){
    delta.assign(U.size(),Vec4{});
    auto blockDiag=[](double v){BlockThomas5::B b{};for(int k=0;k<5;k++)BlockThomas5::at(b,k,k)=v;return b;};
    auto addLine=[&](bool xdir){
      int nline=xdir?mesh.ny:mesh.nx, ncell=xdir?mesh.nx:mesh.ny;
      for(int line=0;line<nline;line++){
        std::vector<BlockThomas5::B>A(ncell),D(ncell),C(ncell);std::vector<Vec5> b(ncell);
        for(int pos=0;pos<ncell;pos++){
          int cell=xdir?mesh.id(pos,line):mesh.id(line,pos);const auto&cc=mesh.cells[cell];
          Prim q=primitive(U[cell],c.gamma);double dti=dt[cell], al=0.6*dti;
          int fm=xdir?cc.face[0]:cc.face[2], fp=xdir?cc.face[1]:cc.face[3];
          double lm=wavespeed(q,mesh.faces[fm].nx,mesh.faces[fm].ny,c.gamma)*mesh.faces[fm].length/cc.area;
          double lp=wavespeed(q,mesh.faces[fp].nx,mesh.faces[fp].ny,c.gamma)*mesh.faces[fp].length/cc.area;
          D[pos]=blockDiag(1.0+al*(lm+lp)); if(pos>0)A[pos]=blockDiag(-al*lm); if(pos+1<ncell)C[pos]=blockDiag(-al*lp);
          b[pos].fill(0);for(int k=0;k<4;k++)b[pos][k]=-dti*R[cell][k]/cc.area;
        }
        if(!BlockThomas5::solve(A,D,C,b)) return false;
        for(int pos=0;pos<ncell;pos++){int cell=xdir?mesh.id(pos,line):mesh.id(line,pos);for(int k=0;k<4;k++)delta[cell][k]+=b[pos][k];}
      }
      return true;
    };
    return addLine(true)&&addLine(false);
  }
  bool update(){
    const int nc=static_cast<int>(U.size());
#ifdef NOZZLE_OPENMP
#pragma omp parallel for if(parallel()) schedule(static)
#endif
    for(int n=0;n<nc;n++){const auto&cc=mesh.cells[n];double s=0;Prim q=primitive(U[n],c.gamma);for(int fi:cc.face){const Face&f=mesh.faces[fi];s+=wavespeed(q,f.nx,f.ny,c.gamma)*f.length;}dt[n]=c.cfl*cc.area/std::max(s,1e-12);}
    std::vector<Vec4> trial=U,delta; if(c.solver=="implicit"){
      if(!directionalCorrection(delta)) return false;
      double alpha=1.0;for(int retry=0;retry<12;retry++){bool ok=true;for(size_t n=0;n<U.size();n++){trial[n]=add(U[n],delta[n],alpha);if(!valid(trial[n],c.gamma)){ok=false;break;}}if(ok){U.swap(trial);return true;}alpha*=.5;}return false;
    }
    double alpha=1;for(int retry=0;retry<12;retry++){bool ok=true;for(size_t n=0;n<U.size();n++){trial[n]=add(U[n],R[n],-alpha*dt[n]/mesh.cells[n].area);if(!valid(trial[n],c.gamma)){ok=false;break;}}if(ok){U.swap(trial);return true;}alpha*=.5;}return false;
  }
  double residualNorm()const{double s=0;for(size_t n=0;n<U.size();n++)for(int k=0;k<4;k++){double z=R[n][k]/mesh.cells[n].area;s+=z*z;}return std::sqrt(s/U.size());}
  double massAt(int boundary)const{double m=0;int count=0;for(const auto&f:mesh.faces)if(f.bc==boundary){Vec4 L=U[f.left],G=ghost(f,L);auto F=roeFlux(L,G,f.nx,f.ny,c.gamma);m+=F[0]*f.length;count++;}return count?m:0;}
  ExitSample exitSample(double outlet_mass) const {
    ExitSample s; s.mass=outlet_mass; double weight=0;
    for(const auto&f:mesh.faces)if(f.bc==2){Prim q=primitive(U[f.left],c.gamma);weight+=f.length;s.rho+=q.rho*f.length;s.u+=q.u*f.length;s.v+=q.v*f.length;s.p+=q.p*f.length;}
    if(weight>0){s.rho/=weight;s.u/=weight;s.v/=weight;s.p/=weight;} return s;
  }
  static double relativeChange(double a,double b){return std::abs(a-b)/std::max({1e-12,std::abs(a),std::abs(b)});}
  double windowChange(const ExitSample&a,const ExitSample&b) const {double c0=std::sqrt(c.gamma*c.gas_R*c.T0);return std::max({relativeChange(a.mass,b.mass),relativeChange(a.rho,b.rho),relativeChange(a.u,b.u),std::abs(a.v-b.v)/c0,relativeChange(a.p,b.p)});}
  int run(){
    fs::create_directories(c.output); hist.open(c.output+"/residual.csv");
    hist<<std::setprecision(17)<<"iter,residual,normalized,inlet_mass,outlet_mass,mass_imbalance,window_change,min_rho,min_p\n";
    double ref=0,final_norm=0,final_imbalance=std::numeric_limits<double>::infinity(),final_stability=std::numeric_limits<double>::infinity();
    std::deque<ExitSample> window; auto start=std::chrono::steady_clock::now(); int status=2,last_iter=0;
    for(int it=0;it<c.max_iter;it++){
      last_iter=it; residual(); double rn=residualNorm(); if(it==0)ref=std::max(rn,1e-12); final_norm=rn/ref;
      double mr=1e9,mp=1e9;for(auto&u:U){auto q=primitive(u,c.gamma);mr=std::min(mr,q.rho);mp=std::min(mp,q.p);}
      double mi=massAt(1),mo=massAt(2); final_imbalance=std::abs(mi+mo)/std::max(1e-12,std::abs(mi));
      window.push_back(exitSample(mo)); if(static_cast<int>(window.size())>c.stability_window)window.pop_front();
      final_stability=static_cast<int>(window.size())==c.stability_window?windowChange(window.front(),window.back()):std::numeric_limits<double>::infinity();
      hist<<it<<","<<rn<<","<<final_norm<<","<<mi<<","<<mo<<","<<final_imbalance<<","<<final_stability<<","<<mr<<","<<mp<<"\n";
      if(it%c.report_every==0)std::cout<<"iter "<<it<<" residual "<<std::setprecision(6)<<final_norm<<" mass "<<mi<<" "<<mo<<" stability "<<final_stability<<"\n";
      if(final_norm<=c.residual_drop&&final_imbalance<=c.mass_tolerance&&final_stability<=c.stability_tolerance){status=0;break;}
      if(!update()){status=3;break;}
    }
    auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); std::ofstream meta(c.output+"/run_status.txt");
    meta<<std::setprecision(17)<<"status="<<(status==0?"converged":status==2?"max_iter":"invalid_state")<<"\nseconds="<<sec<<"\niterations="<<last_iter+1<<"\nsolver="<<c.solver<<"\nbackend="<<c.backend<<"\nflux="<<c.flux<<"\nreconstruction="<<c.reconstruction<<"\nnormalized_residual="<<final_norm<<"\nmass_imbalance="<<final_imbalance<<"\nwindow_change="<<final_stability<<"\n";
    writeField(); return status;
  }
  void writeField()const{std::ofstream out(c.output+"/field.csv");out<<std::setprecision(17)<<"i,j,x,y,rho,u,v,p,M\n";for(int j=0;j<mesh.ny;j++)for(int i=0;i<mesh.nx;i++){int n=mesh.id(i,j);auto q=primitive(U[n],c.gamma);out<<i<<","<<j<<","<<mesh.cells[n].xc<<","<<mesh.cells[n].yc<<","<<q.rho<<","<<q.u<<","<<q.v<<","<<q.p<<","<<std::hypot(q.u,q.v)/std::sqrt(c.gamma*q.p/q.rho)<<"\n";}}
};

static bool selfTest(){
  auto near=[](double a,double b,double tol=1e-12){return std::abs(a-b)<=tol*std::max({1.0,std::abs(a),std::abs(b)});};
  auto checkQuad=[&](const nozzle_geometry::Point2&a,const nozzle_geometry::Point2&b,const nozzle_geometry::Point2&c,const nozzle_geometry::Point2&d,double area,double cx,double cy){const auto g=nozzle_geometry::quadrilateral(a,b,c,d);return near(g.signed_area,area)&&near(g.centroid_x,cx)&&near(g.centroid_y,cy);};
  if(!checkQuad({0,0},{2,0},{2,1},{0,1},2,1,.5)||
     !checkQuad({0,0},{2,0},{3,1},{1,1},2,1.5,.5)||
     !checkQuad({0,0},{3,0},{2,2},{0,2},5,19.0/15.0,14.0/15.0)){
    std::cerr<<"quadrilateral area/centroid regression failed\n";return false;
  }
  if(!(nozzle_geometry::quadrilateral({0,0},{0,2},{2,2},{3,0}).signed_area<0.0)){
    std::cerr<<"signed-area orientation regression failed\n";return false;
  }
  Mesh m;m.build(8,4,10,.5,1.5);if(!m.geometryOk()){std::cerr<<"geometry failed\n";return false;}
  double measured_area=0.0,expected_area=0.0,dx=m.L/m.nx;
  for(int j=0;j<m.ny;j++)for(int i=0;i<m.nx;i++){const double expected=dx*(m.h(dx*i)+m.h(dx*(i+1)))/m.ny;const double measured=m.cells[m.id(i,j)].area;if(!near(measured,expected)){std::cerr<<"nozzle cell area mismatch at ("<<i<<","<<j<<"): "<<measured<<" vs "<<expected<<"\n";return false;}measured_area+=measured;expected_area+=expected;}
  if(!near(measured_area,expected_area)){std::cerr<<"nozzle total area mismatch\n";return false;}
  Prim q{1.2,.3,.1,1.1};auto u=conservative(q,1.4);auto q2=primitive(u,1.4);for(double x:{q2.rho-q.rho,q2.u-q.u,q2.v-q.v,q2.p-q.p})if(std::abs(x)>1e-12)return false;auto f=rusanov(u,u,1,0,1.4);auto exact=physicalFlux(q,1,0,1.4);for(int k=0;k<4;k++)if(std::abs(f[k]-exact[k])>1e-12)return false;
  Mesh straight;straight.build(8,4,10,.5,1.0);std::vector<Vec4> rr(straight.cells.size(),Vec4{});Prim uq{1.2,.3,0,1.1};auto uu=conservative(uq,1.4);for(const auto&face:straight.faces){Vec4 L=uu,RR=uu;auto ff=rusanov(L,RR,face.nx,face.ny,1.4);if(face.left>=0)for(int k=0;k<4;k++)rr[face.left][k]+=ff[k]*face.length;if(face.right>=0)for(int k=0;k<4;k++)rr[face.right][k]-=ff[k]*face.length;}double maxr=0;for(const auto&r:rr)for(double z:r)maxr=std::max(maxr,std::abs(z));if(maxr>1e-12){std::cerr<<"uniform straight-channel residual="<<maxr<<"\n";return false;}
  std::cout<<"self-test: signed quadrilateral geometry, exact nozzle-cell area, face closure, state conversion, equal-state flux, straight-channel uniform flow PASS\n";return true;}
static bool linearTest(){
  int n=4;std::vector<BlockThomas5::B>A(n),D(n),C(n);std::vector<Vec5>b(n),rhs0;
  for(int i=0;i<n;i++){A[i].fill(0);D[i].fill(0);C[i].fill(0);for(int k=0;k<5;k++){BlockThomas5::at(D[i],k,k)=3; b[i][k]=1+i+k;}if(i)for(int k=0;k<5;k++)BlockThomas5::at(A[i],k,k)=-.2;if(i+1<n)for(int k=0;k<5;k++)BlockThomas5::at(C[i],k,k)=-.1;}
  rhs0=b;if(!BlockThomas5::solve(A,D,C,b))return false;double err=0;
  for(int i=0;i<n;i++)for(int r=0;r<5;r++){double lhs=0;for(int k=0;k<5;k++){if(i)lhs+=BlockThomas5::at(A[i],r,k)*b[i-1][k];lhs+=BlockThomas5::at(D[i],r,k)*b[i][k];if(i+1<n)lhs+=BlockThomas5::at(C[i],r,k)*b[i+1][k];}err=std::max(err,std::abs(lhs-rhs0[i][r]));}
  if(err>1e-11){
    std::cerr<<"block-linear residual="<<err<<" sol0="<<b[0][0]<<"\n";
    return false;
  }
  std::cout<<"block-linear Thomas PASS (n="<<n<<", block=5x5, max_residual="<<err<<")\n";
  return true;
}

int main(int argc,char**argv){try{if(argc>1&&std::string(argv[1])=="--self-test")return selfTest()?0:1;if(argc>1&&std::string(argv[1])=="--linear-test")return linearTest()?0:1;if(argc>3)throw std::runtime_error("usage: nozzle_solver [CONFIG] [OUTPUT_DIR]");std::string cfg=argc>1?argv[1]:"cases/subsonic.cfg";Config c=loadConfig(cfg);if(argc>2)c.output=argv[2];Solver s(c);if(!s.mesh.geometryOk()){std::cerr<<"geometry check failed\n";return 4;}s.initialize();std::cout<<"case="<<c.name<<" grid="<<c.nx<<"x"<<c.ny<<" flux="<<c.flux<<" reconstruction="<<c.reconstruction<<" solver="<<c.solver<<"\n";return s.run();}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 5;}}
