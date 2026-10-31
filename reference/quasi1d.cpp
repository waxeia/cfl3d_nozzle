#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

static double ar(double M,double g){return (1/M)*std::pow(2/(g+1)*(1+(g-1)*.5*M*M),(g+1)/(2*(g-1)));}
static double root(double ratio,double g,bool sup){double lo=sup?1.0+1e-10:1e-8,hi=sup?20.0:1.0-1e-10;for(int n=0;n<120;n++){double m=.5*(lo+hi),v=ar(m,g);if(sup?(v<ratio):(v>ratio))lo=m;else hi=m;}return .5*(lo+hi);}
int main(int argc,char**argv){std::string branch=argc>1?argv[1]:"subsonic",path=argc>2?argv[2]:"quasi1d.csv";int nx=argc>3?std::stoi(argv[3]):256;double g=1.4,L=10,ht=.5,rA=1.5,p0=1,T0=1,Astar;
  if(branch=="choked")Astar=2*ht;else{double pb=.92;double Me=std::sqrt(2/(g-1)*(std::pow(1/pb,(g-1)/g)-1));Astar=2*ht*rA/ar(Me,g);}std::ofstream o(path);o<<"x,area,M,rho,u,p,T,mass_flow,branch\n";
  o<<std::setprecision(17);
  for(int i=0;i<=nx;i++){double x=L*i/nx,z=(x-L/2)/(L/2),A=2*ht*(1+(rA-1)*z*z),ratio=A/Astar;bool sup=branch=="choked"&&x>L/2;double M=(std::abs(ratio-1)<1e-10)?1:root(ratio,g,sup),T=T0/(1+.5*(g-1)*M*M),p=p0*std::pow(T/T0,g/(g-1)),rho=p/T,u=M*std::sqrt(g*T);o<<x<<","<<A<<","<<M<<","<<rho<<","<<u<<","<<p<<","<<T<<","<<rho*u*A<<","<<(sup?"supersonic":"subsonic")<<"\n";}
  std::cout<<"wrote "<<path<<" using explicit "<<branch<<" branch intervals\n";
}
