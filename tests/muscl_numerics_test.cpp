#define NOZZLE_NO_MAIN
#include "../src/nozzle_solver.cpp"
#include "../src/explicit_backend_common.hpp"

static void require(bool condition,const std::string& message){
  if(!condition)throw std::runtime_error(message);
}
static double maxDiff(const Vec4&a,const Vec4&b){
  double e=0;for(int k=0;k<4;k++)e=std::max(e,std::abs(a[k]-b[k]));return e;
}
static void fluxTests(){
  double worst=0;
  for(double angle:{0.0,.37,1.2,2.7}){
    double nx=std::cos(angle),ny=std::sin(angle);
    auto q=[&](double rho,double un,double ut,double p){return Prim{rho,un*nx-ut*ny,un*ny+ut*nx,p};};
    // Contact, shear, and fully supersonic Riemann problems have exact upwind
    // fluxes. Equal-state tests alone cannot see wrong Roe eigenvectors.
    for(const auto& pair:std::vector<std::pair<Prim,Prim>>{
          {q(1,.7,.3,1),q(1.6,.7,.3,1)},
          {q(1,.7,.2,1),q(1,.7,.6,1)},
          {q(1,4,.2,1),q(.8,3.8,.5,.7)}}){
      auto L=conservative(pair.first,1.4),R=conservative(pair.second,1.4);
      auto f=roeFlux(L,R,nx,ny,1.4),exact=physicalFlux(pair.first,nx,ny,1.4);
      worst=std::max(worst,maxDiff(f,exact));
      require(maxDiff(f,exact)<1e-12,"Roe contact/shear/supersonic flux");
      auto reverse=roeFlux(R,L,-nx,-ny,1.4);
      for(int k=0;k<4;k++)require(std::abs(f[k]+reverse[k])<1e-12,"oriented flux conservation");
      double backend[4];dRoe(L.data(),R.data(),nx,ny,1.4,backend);
      for(int k=0;k<4;k++)require(std::abs(f[k]-backend[k])<1e-12,"shared backend Roe formula");
    }
  }
  std::cout<<"Roe exact contact/shear/supersonic flux max_error="<<worst<<"\n";
}
static void reconstructionTests(){
  Config c;c.nx=16;c.ny=8;c.reconstruction="muscl";Solver s(c);
  auto exact=[](double x,double y){return Vec4{2+.03*x+.07*y,.3+.01*x-.02*y,.04-.003*x+.02*y,1+.01*x+.04*y};};
  for(size_t n=0;n<s.U.size();n++)s.U[n]=fromPrimitiveArray(exact(s.mesh.cells[n].xc,s.mesh.cells[n].yc),c.gamma);
  s.computeSlopes();double error=0;
  for(size_t n=0;n<s.U.size();n++)for(int fi:s.mesh.cells[n].face){
    auto&f=s.mesh.faces[fi];auto q=primitiveArray(s.faceState(n,f),c.gamma);
    error=std::max(error,maxDiff(q,exact(f.mx,f.my)));
  }
  require(error<2e-12,"affine exactness on curved mesh including boundary faces");
  require(s.active_cells==int(s.U.size())&&s.positivity_cells==0,"MUSCL must be active");
  std::cout<<"Curved-mesh affine face reconstruction max_error="<<error<<"\n";
  // Force a boundary extrapolation to violate positivity before scaling.
  for(int j=0;j<c.ny;j++)for(int i=0;i<c.nx;i++)s.U[s.mesh.id(i,j)]=conservative({i==0?1e-5:1.,.3,0,i==0?1e-5:1.},c.gamma);
  s.computeSlopes();require(s.positivity_cells==c.ny,"localized positivity protection");
  for(size_t n=0;n<s.U.size();n++)for(int fi:s.mesh.cells[n].face)
    require(valid(s.faceState(n,s.mesh.faces[fi]),c.gamma),"positive protected face state");
  std::cout<<"Forced positivity test: protected "<<s.positivity_cells<<"/"<<s.U.size()<<" cells; min_theta="<<s.min_theta<<"\n";
  for(auto& state:s.U)state=conservative({1.5e-10,.3,0,1.5e-10},c.gamma);
  s.computeSlopes();require(s.min_theta==1,"valid dilute constant state must not produce negative theta");
  // Straight-channel uniform flow through the complete solver path.
  c.area_ratio=1;c.init="quasi1d";c.outlet_pressure=.92;c.time_integrator="ssprk3";c.cfl=.8;
  Solver uniform(c);uniform.initialize();const auto before=uniform.U;
  for(int step=0;step<100;step++){
    uniform.residual();require(uniform.residualNorm()<2e-12,"uniform flow residual");
    require(uniform.update(),"uniform flow SSPRK3 update");
  }
  for(size_t n=0;n<before.size();n++)require(maxDiff(before[n],uniform.U[n])<2e-12,"uniform flow drift");
  // Numerical flux accounting must close the domain's mass balance exactly.
  s.initialize();s.residual();double total=0;for(auto&r:s.R)total+=r[0];
  double boundary=s.massAt(1)+s.massAt(2)+s.massAt(3)+s.massAt(4);
  require(std::abs(total-boundary)<2e-12,"residual and recorded mass use identical face fluxes");
  require(std::abs(s.massAt(3))+std::abs(s.massAt(4))<2e-12,"slip wall mass flux");
  c.area_ratio=1.5;c.init="quasi1d";c.cfl=100;
  Solver retry(c);retry.initialize();retry.residual();
  require(retry.update()&&retry.step_retries>0,"oversized step must retry from the old state");
  c.cfl=std::ldexp(c.cfl,-int(retry.step_retries));
  Solver smallStep(c);smallStep.initialize();smallStep.residual();
  require(smallStep.update(),"matching small step");
  for(size_t n=0;n<retry.U.size();n++)require(maxDiff(retry.U[n],smallStep.U[n])<1e-12,"SSPRK3 retry rollback");
}

// Independent exact Euler solution: a periodic smooth entropy wave,
// rho=1+0.2*sin(2*pi*(x-u*t)), p=1, u=0.7, v=0. Cell averages are analytic.
// Only boundary topology is replaced; reconstruction and flux use production
// functions. Global, small time steps isolate spatial error from pseudo-time.
static void periodicResidual(Solver&s){
  s.residual();
  for(int j=0;j<s.mesh.ny;j++){
    int first=j,last=s.mesh.nx*s.mesh.ny+j;
    const auto& leftFace=s.mesh.faces[first];const auto& rightFace=s.mesh.faces[last];
    auto L=s.faceState(rightFace.left,rightFace),R=s.faceState(leftFace.left,leftFace);
    auto F=roeFlux(L,R,1,0,s.c.gamma);
    for(int k=0;k<4;k++){
      s.R[leftFace.left][k]+=(-F[k]-s.face_flux[first][k])*leftFace.length;
      s.R[rightFace.left][k]+=(F[k]-s.face_flux[last][k])*rightFace.length;
      s.face_flux[first][k]=-F[k];s.face_flux[last][k]=F[k];
    }
  }
}
static void accuracyStudy(const std::string& directory,double timeCfl=.12){
  fs::create_directories(directory);std::ofstream out(directory+"/entropy_wave.csv");
  out<<std::setprecision(17)<<"reconstruction,nx,L1_rho,Linf_rho,L1_order,mass_error,pressure_error\n";
  const double pi=std::acos(-1.0),velocity=.7,finalTime=1/velocity;
  for(const std::string reconstruction:{"first","muscl"}){
    double previous=0;
    for(int nx:{32,64,128,256}){
      Config c;c.nx=nx;c.ny=2;c.length=1;c.area_ratio=1;c.reconstruction=reconstruction;
      Solver s(c);double dx=1.0/nx,averageFactor=std::sin(pi*dx)/(pi*dx);
      auto exact=[&](double x){return 1+.2*averageFactor*std::sin(2*pi*x);};
      for(size_t n=0;n<s.U.size();n++)s.U[n]=conservative({exact(s.mesh.cells[n].xc),velocity,0,1},c.gamma);
      int steps=int(std::ceil(finalTime/(timeCfl*dx/(velocity+std::sqrt(1.4/.8)))));
      double dt=finalTime/steps;
      for(int it=0;it<steps;it++){
        auto old=s.U;
        for(int stage=0;stage<3;stage++){
          periodicResidual(s);double w=stage==1?.75:stage==2?1.0/3:0;
          for(size_t n=0;n<s.U.size();n++)for(int k=0;k<4;k++)
            s.U[n][k]=w*old[n][k]+(1-w)*(s.U[n][k]-dt*s.R[n][k]/s.mesh.cells[n].area);
        }
      }
      double l1=0,linf=0,mass=0,pressure=0;
      for(size_t n=0;n<s.U.size();n++){
        auto q=primitive(s.U[n],c.gamma);double e=std::abs(q.rho-exact(s.mesh.cells[n].xc));
        l1+=e*s.mesh.cells[n].area;linf=std::max(linf,e);mass+=q.rho*s.mesh.cells[n].area;pressure=std::max(pressure,std::abs(q.p-1));
      }
      double order=previous?std::log2(previous/l1):0;
      out<<reconstruction<<","<<nx<<","<<l1<<","<<linf<<","<<order<<","<<std::abs(mass-1)<<","<<pressure<<"\n";
      std::cout<<reconstruction<<" nx="<<nx<<" L1="<<l1<<" order="<<order<<"\n";
      require(std::abs(mass-1)<2e-11&&pressure<2e-11,"entropy wave invariants");
      if(nx==256)require(order>(reconstruction=="muscl"?1.7:.8),"independent spatial accuracy regression");
      previous=l1;
    }
  }
  std::ofstream curved(directory+"/curved_reconstruction.csv");
  curved<<std::setprecision(17)<<"nx,ny,L1_face_rho_error,L1_order\n";
  double previous=0;
  for(int nx:{16,32,64,128}){
    Config c;c.nx=nx;c.ny=nx/4;c.reconstruction="muscl";Solver s(c);
    auto rho=[](double x,double y){return 2+.04*x+.005*x*x+.1*y+.06*y*y+.01*x*y;};
    // Degree-2 exact triangle midpoint quadrature gives actual CELL AVERAGES.
    auto tri=[&](auto a,auto b,auto d){
      double area=.5*std::abs((b[0]-a[0])*(d[1]-a[1])-(b[1]-a[1])*(d[0]-a[0]));
      return area*(rho((a[0]+b[0])/2,(a[1]+b[1])/2)+rho((a[0]+d[0])/2,(a[1]+d[1])/2)+rho((b[0]+d[0])/2,(b[1]+d[1])/2))/3;
    };
    for(int j=0;j<c.ny;j++)for(int i=0;i<c.nx;i++){
      auto a=s.mesh.node(i,j),b=s.mesh.node(i+1,j),d=s.mesh.node(i,j+1),e=s.mesh.node(i+1,j+1);
      int n=s.mesh.id(i,j);double mean=(tri(a,b,e)+tri(a,e,d))/s.mesh.cells[n].area;
      s.U[n]=conservative({mean,.7,0,1},c.gamma);
    }
    s.computeSlopes();double error=0,area=0;
    for(size_t n=0;n<s.U.size();n++){
      area+=s.mesh.cells[n].area;
      for(int fi:s.mesh.cells[n].face){const auto&f=s.mesh.faces[fi];error+=.25*s.mesh.cells[n].area*std::abs(s.faceState(n,f)[0]-rho(f.mx,f.my));}
    }
    error/=area;double order=previous?std::log2(previous/error):0;
    curved<<nx<<","<<c.ny<<","<<error<<","<<order<<"\n";
    std::cout<<"curved cell-average reconstruction nx="<<nx<<" L1="<<error<<" order="<<order<<"\n";
    if(nx==128)require(order>1.9,"curved-grid face reconstruction accuracy");
    previous=error;
  }
}
int main(int argc,char**argv){
  try{
    auto project=fs::path(__FILE__).parent_path().parent_path();
    require(loadConfig((project/"cases/subsonic.cfg").string()).reconstruction=="first","preserve first-order config default");
    if(argc==3&&std::string(argv[1])=="--accuracy-study"){accuracyStudy(argv[2]);return 0;}
    if(argc==3&&std::string(argv[1])=="--accuracy-half-dt"){accuracyStudy(argv[2],.06);return 0;}
    require(selfTest()&&linearTest(),"existing numerical regressions");
    fluxTests();reconstructionTests();std::cout<<"MUSCL numerical regressions PASS\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
