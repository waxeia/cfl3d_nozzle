#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh_geometry.hpp"

#ifndef NOZZLE_INLINE
#define NOZZLE_INLINE inline
#endif

struct BackendConfig {
  int nx=128,ny=32,max_iter=3000,report_every=250,stability_window=200;
  double length=10,throat_half_height=.5,area_ratio=1.5,gamma=1.4,gas_R=1,p0=1,T0=1;
  double outlet_pressure=.88,cfl=.25,residual_drop=1e-6,mass_tolerance=1e-3,stability_tolerance=1e-4,init_mach=.2;
  bool choked=false;
  std::string name="case",init="uniform",flux="roe",reconstruction="first",solver="explicit",output="output";
};
struct DeviceParams {double gamma=1.4,gas_R=1,p0=1,T0=1,outlet_pressure=.88,cfl=.25;};
inline DeviceParams deviceParams(const BackendConfig&c){return {c.gamma,c.gas_R,c.p0,c.T0,c.outlet_pressure,c.cfl};}

inline BackendConfig loadBackendConfig(const std::string&path){
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open config: "+path);
  std::map<std::string,std::string>kv;std::string line;
  auto trim=[](std::string s){const char*ws=" \t\r\n";auto a=s.find_first_not_of(ws);if(a==std::string::npos)return std::string{};s.erase(0,a);s.erase(s.find_last_not_of(ws)+1);return s;};
  while(std::getline(in,line)){auto h=line.find('#');if(h!=std::string::npos)line.resize(h);auto e=line.find('=');if(e!=std::string::npos)kv[trim(line.substr(0,e))]=trim(line.substr(e+1));}
  BackendConfig c;auto si=[&](const char*k,int&v){if(kv.count(k))v=std::stoi(kv[k]);};auto sd=[&](const char*k,double&v){if(kv.count(k))v=std::stod(kv[k]);};auto ss=[&](const char*k,std::string&v){if(kv.count(k))v=kv[k];};
  si("nx",c.nx);si("ny",c.ny);si("max_iter",c.max_iter);si("report_every",c.report_every);si("stability_window",c.stability_window);
  sd("length",c.length);sd("throat_half_height",c.throat_half_height);sd("area_ratio",c.area_ratio);sd("gamma",c.gamma);sd("R",c.gas_R);sd("p0",c.p0);sd("T0",c.T0);sd("outlet_pressure",c.outlet_pressure);sd("cfl",c.cfl);sd("residual_drop",c.residual_drop);sd("mass_tolerance",c.mass_tolerance);sd("stability_tolerance",c.stability_tolerance);sd("init_mach",c.init_mach);
  ss("name",c.name);ss("init",c.init);ss("flux",c.flux);ss("reconstruction",c.reconstruction);ss("solver",c.solver);ss("output",c.output);
  if(kv.count("choked"))c.choked=kv["choked"]=="true"||kv["choked"]=="1";
  for(double value:{c.length,c.throat_half_height,c.area_ratio,c.gamma,c.gas_R,c.p0,c.T0,c.outlet_pressure,c.cfl,c.residual_drop,c.mass_tolerance,c.stability_tolerance,c.init_mach})if(!std::isfinite(value))throw std::runtime_error("configuration contains a non-finite number");
  if(c.nx<2||c.ny<2||c.max_iter<1||c.report_every<1||c.length<=0.0||c.throat_half_height<=0.0||c.area_ratio<=0.0||c.gamma<=1.0||c.gas_R<=0.0||c.p0<=0.0||c.T0<=0.0||c.outlet_pressure<=0.0||c.cfl<=0.0||c.residual_drop<=0.0||c.mass_tolerance<=0.0||c.stability_tolerance<=0.0||c.init_mach<0.0||c.stability_window<1)throw std::runtime_error("invalid grid, geometry, gas, or convergence setting");
  if(c.solver!="explicit"||c.reconstruction!="first"||c.flux!="roe")throw std::runtime_error("device backends currently require solver=explicit, reconstruction=first, flux=roe");
  if(kv.count("time_integrator")&&kv["time_integrator"]!="euler")throw std::runtime_error("device backends currently require time_integrator=euler");
  if(c.init!="uniform"&&c.init!="quasi1d")throw std::runtime_error("init must be uniform or quasi1d");
  return c;
}

struct DeviceFace {int left=-1,right=-1,bc=0;double nx=0,ny=0,length=0;};
struct DeviceCell {double area=0,xc=0,yc=0;int face[4]{};};
struct BackendMesh {
  int nx=0,ny=0;std::vector<DeviceFace>faces;std::vector<DeviceCell>cells;
  int id(int i,int j)const{return j*nx+i;}
  void build(const BackendConfig&c){
    if(c.nx<2||c.ny<2||c.length<=0.0||c.throat_half_height<=0.0||c.area_ratio<=0.0)throw std::runtime_error("invalid mesh dimensions or nozzle geometry");
    nx=c.nx;ny=c.ny;std::vector<nozzle_geometry::Point2>nodes((nx+1)*(ny+1));
    auto h=[&](double x){double z=(x-c.length/2)/(c.length/2);return c.throat_half_height*(1+(c.area_ratio-1)*z*z);};
    for(int j=0;j<=ny;j++){double eta=-1+2.0*j/ny;for(int i=0;i<=nx;i++){double x=c.length*i/nx;nodes[j*(nx+1)+i]={x,eta*h(x)};}}
    cells.assign(nx*ny,{});faces.clear();faces.reserve((nx+1)*ny+nx*(ny+1));
    auto edge=[&](int a,int b,int bc){auto A=nodes[a],B=nodes[b];double dx=B[0]-A[0],dy=B[1]-A[1],len=std::hypot(dx,dy);if(!(len>0.0))throw std::runtime_error("zero-length mesh face");DeviceFace f;f.bc=bc;f.nx=dy/len;f.ny=-dx/len;f.length=len;return f;};
    for(int i=0;i<=nx;i++)for(int j=0;j<ny;j++){DeviceFace f=edge(j*(nx+1)+i,(j+1)*(nx+1)+i,i==0?1:(i==nx?2:0));if(i>0)f.left=id(i-1,j);if(i<nx)f.right=id(i,j);if(i==0){f.left=id(0,j);f.right=-1;f.nx=-f.nx;f.ny=-f.ny;}if(i==nx)f.left=id(nx-1,j);int fi=static_cast<int>(faces.size());faces.push_back(f);if(i>0)cells[id(i-1,j)].face[1]=fi;else cells[id(0,j)].face[0]=fi;if(i<nx)cells[id(i,j)].face[0]=fi;}
    for(int i=0;i<nx;i++)for(int j=0;j<=ny;j++){DeviceFace f=edge(j*(nx+1)+i,j*(nx+1)+i+1,j==0?3:(j==ny?4:0));if(j<ny)f.left=id(i,j);if(j>0)f.right=id(i,j-1);if(j==0){f.left=id(i,0);f.right=-1;}if(j==ny){f.left=id(i,ny-1);f.right=-1;f.nx=-f.nx;f.ny=-f.ny;}int fi=static_cast<int>(faces.size());faces.push_back(f);if(j>0)cells[id(i,j-1)].face[3]=fi;else cells[id(i,0)].face[2]=fi;if(j<ny)cells[id(i,j)].face[2]=fi;else cells[id(i,ny-1)].face[3]=fi;}
    for(int j=0;j<ny;j++)for(int i=0;i<nx;i++){
      const auto a=nodes[j*(nx+1)+i],b=nodes[j*(nx+1)+i+1],d=nodes[(j+1)*(nx+1)+i],e=nodes[(j+1)*(nx+1)+i+1];
      const auto geometry=nozzle_geometry::quadrilateral(a,b,e,d);
      if(!(geometry.signed_area>0.0)||!std::isfinite(geometry.signed_area)||!std::isfinite(geometry.centroid_x)||!std::isfinite(geometry.centroid_y))throw std::runtime_error("non-positive or invalid cell geometry at ("+std::to_string(i)+","+std::to_string(j)+")");
      auto&cc=cells[id(i,j)];cc.area=geometry.signed_area;cc.xc=geometry.centroid_x;cc.yc=geometry.centroid_y;
    }
  }
  bool geometryOk(double tol=1e-10)const{
    for(std::size_t n=0;n<cells.size();++n){const auto&c=cells[n];if(!(c.area>0.0)||!std::isfinite(c.area)||!std::isfinite(c.xc)||!std::isfinite(c.yc))return false;double sx=0,sy=0;for(int fi:c.face){if(fi<0||fi>=static_cast<int>(faces.size()))return false;const auto&f=faces[fi];if(!(f.length>0.0)||!std::isfinite(f.length)||std::abs(std::hypot(f.nx,f.ny)-1.0)>tol)return false;double s=0;if(f.left==static_cast<int>(n))s=1;else if(f.right==static_cast<int>(n))s=-1;else return false;sx+=s*f.nx*f.length;sy+=s*f.ny*f.length;}if(std::hypot(sx,sy)>tol)return false;}return true;
  }
};

struct DPrim {double rho,u,v,p;};
NOZZLE_INLINE DPrim dPrimitive(const double*U,double g){DPrim q{U[0],U[1]/U[0],U[2]/U[0],0};q.p=(g-1)*(U[3]-.5*q.rho*(q.u*q.u+q.v*q.v));return q;}
NOZZLE_INLINE void dConservative(const DPrim&q,double g,double*U){U[0]=q.rho;U[1]=q.rho*q.u;U[2]=q.rho*q.v;U[3]=q.p/(g-1)+.5*q.rho*(q.u*q.u+q.v*q.v);}
NOZZLE_INLINE double dWave(const DPrim&q,double nx,double ny,double g){return fabs(q.u*nx+q.v*ny)+sqrt(g*q.p/q.rho);}
NOZZLE_INLINE void dPhysical(const DPrim&q,double nx,double ny,double g,double*F){double un=q.u*nx+q.v*ny,E=q.p/(g-1)+.5*q.rho*(q.u*q.u+q.v*q.v);F[0]=q.rho*un;F[1]=q.rho*q.u*un+q.p*nx;F[2]=q.rho*q.v*un+q.p*ny;F[3]=(E+q.p)*un;}
NOZZLE_INLINE void dGhost(const DeviceFace&f,const double*inside,const DeviceParams&c,double*out){
  DPrim q=dPrimitive(inside,c.gamma);if(f.bc==3||f.bc==4){double un=q.u*f.nx+q.v*f.ny;q.u-=2*un*f.nx;q.v-=2*un*f.ny;dConservative(q,c.gamma,out);return;}
  if(f.bc==1){double ci=sqrt(c.gamma*q.p/q.rho),Jm=q.u-2*ci/(c.gamma-1),a0sq=c.gamma*c.gas_R*c.T0,lo=0,hi=sqrt(2*a0sq/(c.gamma+1));for(int n=0;n<80;n++){double mid=.5*(lo+hi),a=sqrt(fmax(1e-12,a0sq-.5*(c.gamma-1)*mid*mid));if(mid-2*a/(c.gamma-1)-Jm>0)hi=mid;else lo=mid;}double u=.5*(lo+hi),T=c.T0-u*u*(c.gamma-1)/(2*c.gamma*c.gas_R),p=c.p0*pow(T/c.T0,c.gamma/(c.gamma-1)),rho=p/(c.gas_R*T);dConservative(DPrim{rho,u,0,p},c.gamma,out);return;}
  if(f.bc==2){double ci=sqrt(c.gamma*q.p/q.rho);if(q.u>ci){for(int k=0;k<4;k++)out[k]=inside[k];return;}double entropy=q.p/pow(q.rho,c.gamma),rho=pow(c.outlet_pressure/entropy,1/c.gamma),co=sqrt(c.gamma*c.outlet_pressure/rho),u=q.u+2*(ci-co)/(c.gamma-1);dConservative(DPrim{rho,u,q.v,c.outlet_pressure},c.gamma,out);return;}
  for(int k=0;k<4;k++)out[k]=inside[k];
}
NOZZLE_INLINE void dRoe(const double*UL,const double*UR,double nx,double ny,double g,double*out){
  DPrim l=dPrimitive(UL,g),r=dPrimitive(UR,g);double fl[4],fr[4];dPhysical(l,nx,ny,g,fl);dPhysical(r,nx,ny,g,fr);
  double sl=sqrt(l.rho),sr=sqrt(r.rho),den=sl+sr,u=(sl*l.u+sr*r.u)/den,v=(sl*l.v+sr*r.v)/den,Hl=(UL[3]+l.p)/l.rho,Hr=(UR[3]+r.p)/r.rho,H=(sl*Hl+sr*Hr)/den,un=u*nx+v*ny,ut=-u*ny+v*nx,q2=u*u+v*v,c=sqrt(fmax(1e-12,(g-1)*(H-.5*q2))),rho=sl*sr;
  double dr=r.rho-l.rho,dp=r.p-l.p,dun=(r.u-l.u)*nx+(r.v-l.v)*ny,dut=-(r.u-l.u)*ny+(r.v-l.v)*nx,cc=c*c,a[4]={(dp-rho*c*dun)/(2*cc),dr-dp/cc,rho*dut,(dp+rho*c*dun)/(2*cc)},lam[4]={un-c,un,un,un+c};for(int m=0;m<4;m++){double d=.1*c;if(fabs(lam[m])<d)lam[m]=.5*(lam[m]*lam[m]/d+d);lam[m]=fabs(lam[m]);}
  double ev[4][4]={{1,u-c*nx,v-c*ny,H-c*un},{1,u,v,.5*q2},{0,-ny,nx,ut},{1,u+c*nx,v+c*ny,H+c*un}};for(int k=0;k<4;k++){double diss=0;for(int m=0;m<4;m++)diss+=lam[m]*a[m]*ev[m][k];out[k]=.5*(fl[k]+fr[k])-.5*diss;}
}

inline double backendAreaMach(double M,double g){return (1/M)*std::pow(2/(g+1)*(1+.5*(g-1)*M*M),(g+1)/(2*(g-1)));}
inline double backendBranchMach(double ratio,double g,bool sup){if(std::abs(ratio-1)<1e-10)return 1;double lo=sup?1+1e-10:1e-8,hi=sup?20:1-1e-10;for(int n=0;n<100;n++){double mid=.5*(lo+hi),v=backendAreaMach(mid,g);if(sup?(v<ratio):(v>ratio))lo=mid;else hi=mid;}return .5*(lo+hi);}
inline std::vector<double> backendInitialize(const BackendConfig&c,const BackendMesh&m){
  std::vector<double>U(m.cells.size()*4);
  if(c.init=="uniform"){const DPrim q{c.p0/(c.gas_R*c.T0),c.init_mach*std::sqrt(c.gamma*c.gas_R*c.T0),0,c.p0};double state[4];dConservative(q,c.gamma,state);for(std::size_t n=0;n<m.cells.size();++n)for(int k=0;k<4;k++)U[4*n+k]=state[k];return U;}
  double Astar=2*c.throat_half_height;if(!c.choked){double Me=std::sqrt(2/(c.gamma-1)*(std::pow(c.p0/c.outlet_pressure,(c.gamma-1)/c.gamma)-1));Astar=2*c.throat_half_height*c.area_ratio/backendAreaMach(Me,c.gamma);}
  for(int j=0;j<m.ny;j++)for(int i=0;i<m.nx;i++){int n=m.id(i,j);double x=m.cells[n].xc,z=(x-c.length*.5)/(c.length*.5),A=2*c.throat_half_height*(1+(c.area_ratio-1)*z*z),M=backendBranchMach(A/Astar,c.gamma,c.choked&&x>c.length*.5),T=c.T0/(1+.5*(c.gamma-1)*M*M),p=c.p0*std::pow(T/c.T0,c.gamma/(c.gamma-1)),rho=p/(c.gas_R*T),u=M*std::sqrt(c.gamma*c.gas_R*T);double q[4];dConservative(DPrim{rho,u,0,p},c.gamma,q);for(int k=0;k<4;k++)U[4*n+k]=q[k];}
  return U;
}
inline void backendWriteField(const std::string&dir,const BackendConfig&c,const BackendMesh&m,const std::vector<double>&U){std::ofstream out(dir+"/field.csv");out<<std::setprecision(17)<<"i,j,x,y,rho,u,v,p,M\n";for(int j=0;j<m.ny;j++)for(int i=0;i<m.nx;i++){int n=m.id(i,j);DPrim q=dPrimitive(&U[4*n],c.gamma);out<<i<<","<<j<<","<<m.cells[n].xc<<","<<m.cells[n].yc<<","<<q.rho<<","<<q.u<<","<<q.v<<","<<q.p<<","<<std::hypot(q.u,q.v)/std::sqrt(c.gamma*q.p/q.rho)<<"\n";}}
inline double backendRelative(double a,double b){return std::abs(a-b)/std::max({1e-12,std::abs(a),std::abs(b)});}
