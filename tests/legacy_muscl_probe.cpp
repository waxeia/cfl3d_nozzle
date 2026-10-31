// Compile separately: this intentionally examines the frozen pre-repair source.
#define main frozen_solver_main
#include "../output/muscl_repair/legacy/nozzle_solver.cpp"
#undef main

int main(int argc,char**argv){
  if(argc!=3)return 1;
  auto c=loadConfig(argv[1]);Solver s(c);std::ifstream field(std::string(argv[2])+"/field.csv");
  std::string line;std::getline(field,line);
  while(std::getline(field,line)){
    std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);
    int i,j;double x,y,m;Prim q;in>>i>>j>>x>>y>>q.rho>>q.u>>q.v>>q.p>>m;
    s.U[s.mesh.id(i,j)]=conservative(q,c.gamma);
  }
  s.residual();int fallback=0;double minrho=1e99,minp=1e99,mi=0,mo=0,wall=0;
  std::array<double,4> component{};
  std::ofstream spatial(std::string(argv[2])+"/audit_residual_map.csv");
  spatial<<std::setprecision(17)<<"i,j,x,y,mass,momentum_x,momentum_y,energy\n";
  for(size_t n=0;n<s.U.size();n++){
    const auto&cc=s.mesh.cells[n];spatial<<n%c.nx<<","<<n/c.nx<<","<<cc.xc<<","<<cc.yc;
    for(int k=0;k<4;k++){double r=s.R[n][k]/cc.area;component[k]+=r*r;spatial<<","<<r;}spatial<<"\n";
    for(int fi:cc.face){const auto&f=s.mesh.faces[fi];auto q=s.U[n];
      for(int k=0;k<4;k++)q[k]+=.5*(s.slopes[n][2*k]*(f.mx-cc.xc)+s.slopes[n][2*k+1]*(f.my-cc.yc));
      fallback+=!valid(q,c.gamma);auto prim=primitive(q,c.gamma);minrho=std::min(minrho,prim.rho);minp=std::min(minp,prim.p);
    }
  }
  for(size_t fi=0;fi<s.mesh.faces.size();fi++){
    const auto&f=s.mesh.faces[fi];double m=s.face_flux[fi][0]*f.length;
    if(f.bc==1)mi+=m;if(f.bc==2)mo+=m;if(f.bc>=3)wall+=m;
  }
  std::ofstream audit(std::string(argv[2])+"/legacy_probe.txt");audit<<std::setprecision(17);
  audit<<"sample=field_after_legacy_final_update\nraw_face_min_rho="<<minrho<<"\nraw_face_min_p="<<minp<<"\nface_fallback_count="<<fallback
       <<"\nactual_inlet_mass="<<mi<<"\nactual_outlet_mass="<<mo<<"\nactual_mass_imbalance="<<std::abs(mi+mo)/std::abs(mi)
       <<"\nlegacy_diagnostic_mass_imbalance="<<std::abs(s.massAt(1)+s.massAt(2))/std::abs(s.massAt(1))<<"\nwall_mass="<<wall<<"\n";
  for(int k=0;k<4;k++)audit<<"residual_component_"<<k<<"="<<std::sqrt(component[k]/s.U.size())<<"\n";
  Prim a{1,.7,.3,1},b{1.6,.7,.3,1};auto f=roeFlux(conservative(a,1.4),conservative(b,1.4),1,0,1.4);auto exact=physicalFlux(a,1,0,1.4);
  audit<<"moving_contact_energy_flux_error="<<f[3]-exact[3]<<"\n";
  b={1,.7,.6,1};f=roeFlux(conservative(a,1.4),conservative(b,1.4),1,0,1.4);
  audit<<"moving_shear_energy_flux_error="<<f[3]-exact[3]<<"\n";
  // Conserved affine rho with constant velocity and pressure is also affine U.
  auto affine=[](double x,double y){return conservative({2+.03*x+.07*y,.3,.04,1},1.4);};
  for(size_t n=0;n<s.U.size();n++)s.U[n]=affine(s.mesh.cells[n].xc,s.mesh.cells[n].yc);
  s.computeSlopes();double err=0;
  for(size_t n=0;n<s.U.size();n++)for(int fi:s.mesh.cells[n].face){auto&f=s.mesh.faces[fi];err=std::max(err,std::abs(s.faceState(n,f)[0]-affine(f.mx,f.my)[0]));}
  audit<<"affine_rho_face_max_error="<<err<<"\n";
}
