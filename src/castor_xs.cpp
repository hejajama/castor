/*
 * Single inclusive cross section calculations using the
 * AmplitudeLib, for Castor kinematics, so total energy is defined
 *
 * Heikki Mäntysaari <heikki.mantysaari@jyu.fi>, 2014-2018
 */

#include <amplitudelib/amplitudelib.hpp>
#include <tools/config.hpp>
#include <tools/tools.hpp>
#include <pdf/cteq.hpp>
#include <pdf/eps09.hpp>
#include <pdf/ugdpdf.hpp>
#include <amplitudelib/single_inclusive.hpp>
#include "gitsha1.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_miser.h>
#include <ctime>
#include <cmath>
#include <unistd.h>
#include <gsl/gsl_rng.h>

using namespace Amplitude;
using namespace std;

const double default_particle_mass = 0.2;
const double castor_min_pseudorapidity = 5.2;
const double castor_max_pseudorapidity = 6.6;
double rapidity_shift = 0;

int INTWORKSPACEDIVISIONS = 5;
const double INTACCURACY = 0.001;
double LOW_PT_CUT = 1;

// Kinematics
// Rapidity from pseudorapidity and pt
double Rapidity(double eta, double pt, double m=default_particle_mass)
{
    return std::log(
                    ( std::sqrt(m*m + std::pow(pt*std::cosh(eta),2)) + pt*std::sinh(eta))
                    / std::sqrt(m*m + pt*pt)
            );
}

// Pseudorapidity from rapidity
double Pseudorapidity(double y, double pt, double m=default_particle_mass)
{
    return acosh( std::exp(-y)*std::sqrt( std::pow(std::exp(2.0*y)-1.0,2)*std::pow(m,4) + 2.0   *(1+std::exp(4.0*y))*std::pow(m*pt,2) + std::pow(1 + std::exp(2.0*y), 2) * std::pow(pt,4) )
                      / (2.0*pt*std::sqrt( m*m + pt*pt))
                      );
}

double JetEnergy(double y, double pt, double m=default_particle_mass)
{
    return 1.0/2.0 * (std::exp(-y) + std::exp(y)) * std::sqrt(m*m + pt*pt);
}

struct inthelper_castor
{
    SingleInclusive *xs;
    double minE;    // energy bin
    double maxE;
    PDF* pdf;
    double sqrts;
    double y;   // rapidity integrated over
    gsl_integration_workspace* workspace;
    double m;
};

double inthelperf_pt(double pt, void* p);
double inthelperf_y(double y, void* p);

double inthelperf_mc(double* vec, size_t dim, void* par);

double inthelperf_dps_mc(double* vec, size_t dim, void* p);
double inthelperf_3ps_mc(double* vec, size_t dim, void* p);

int main(int argc, char* argv[])
{
    std::stringstream infostr;
    infostr << "# Single parton yield   (c) Heikki Mäntysaari <heikki.mantysaari@jyu.fi>, 2011-2019 " << endl;
    infostr << "# Latest git commit " << g_CASTOR_GIT_SHA1 << " local repo " << g_CASTOR_GIT_LOCAL_CHANGES << endl;
    infostr << "# Command: ";
    for (int i=0; i<argc; i++)
        infostr << argv[i] << " ";
    
    cout << infostr.str() << endl;
    
    gsl_set_error_handler(&ErrHandler);
    double binwidth = -1;

    int DPS_N = 2;   // Number of independent particles in DPS

    if ( argc==1 or (string(argv[1])=="-help" or string(argv[1])=="--help")  )
    {
        cout << "-data [bksolutionfile]" << endl;
        cout << "-minE, -maxE: calorimeter energy range" << endl;
        cout << "-Estep: step size for energy values that we compute" << endl;
        cout << "-sqrts center-of-mass energy [GeV]" << endl;
        cout << "-x0 val: set x0 value for the BK solutions (overrides the value in BK file)" << endl;
        cout << "-gsl_ft: use GSL to directly calculate Fourier transform" << endl;
        cout << "-yshift [value, LHC pA=+/-0.465]" << endl;
        cout << "-DPS N: compute DPS contribution for N particle production" << endl;
        cout << "-m mass: set jet mass in GeV " << endl;
        cout << "-minpt cutoff: set min pT cutoff in GeV" << endl;
	cout << "-bin winwidth: set E bin width " << endl;
        return 0;
        
    }

    cout << "# Note to self: it might be good idea to use quark mass as m!" << endl;
    cout << "# Low pt cut: " << LOW_PT_CUT << endl;

    double x0=-1;
    double sqrts=5020;
    Order order=LO;
    PDF* pdf=NULL;
    FragmentationFunction* fragfun=NULL;
    std::string datafile="";
    FT_Method ft_method = ACC_SERIES;
    std::string datafile_probe="";
    bool deuteron=false;
    double ptstep=0.1;
    double minE=-1; double maxE=-1;
    double Estep = -1;
    int mcintpoints=1e4;
    bool dps=false;
    
    double mass = default_particle_mass;

    for (int i=1; i<argc; i++)
    {
        if (string(argv[i])=="-x0")
            x0 = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-data")
            datafile=argv[i+1];
        else if (string(argv[i])=="-sqrts")
            sqrts = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-minE")
            minE  = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-maxE")
            maxE  = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-Estep")
            Estep = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-lo")
            order=LO;
        else if (string(argv[i])=="-nlo")
            order=NLO;
        else if (string(argv[i])=="-pdf")
        {
			if (string(argv[i+1])=="cteq")
				pdf = new CTEQ();
			else if (string(argv[i+1])=="eps09")
            {
				pdf = new EPS09();
				pdf->SetA(StrToInt(argv[i+2]));
			}
            else if (string(argv[i+1])=="ugd")
			{
				AmplitudeLib* ugdn = new AmplitudeLib(string(argv[i+2]));
				pdf = new UGDPDF(ugdn, StrToReal(argv[i+3]));
			}
			else
			{
				cerr << "Unknown PDF type " << argv[i+1] << endl;
				return -1;
			}
		}
        else if (string(argv[i])=="-m")
            mass = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-minpt")
            LOW_PT_CUT = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-gsl_ft")
            ft_method = GSL;
        else if (string(argv[i])=="-yshift")
            rapidity_shift=StrToReal(argv[i+1]);
        else if (string(argv[i])=="-intdivisions")
            INTWORKSPACEDIVISIONS = StrToInt(argv[i+1]);
        else if (string(argv[i])=="-mcintpoints")
            mcintpoints = StrToReal(argv[i+1]);
        else if (string(argv[i])=="-DPS")
        {
            dps=true;
            DPS_N = StrToInt(argv[i+1]);
        }
        else if (string(argv[i])=="-bin")
            binwidth=StrToReal(argv[i+1]);
        else if (string(argv[i]).substr(0,1)=="-" and string(argv[i-1]) != "-yshift")
        {
            cerr << "Unrecoginzed parameter " << argv[i] << endl;
            return -1;
        }
    }
    

    if (binwidth < 0)
         binwidth=10;
    if (Estep < 0)
        Estep = 50;

    
    // Default PDF and FF
    if (pdf==NULL)
        pdf = new CTEQ();
    
    // Read data
    AmplitudeLib N(datafile);

    N.SetFTMethod(ft_method);
    SingleInclusive xs(&N);
    
    
    
    if (x0>0)
    {
        N.SetX0(x0);
    }
   

    pdf->SetOrder(order);
    

    time_t now = time(0);
    string today = ctime(&now);
    
    char *hostname = new char[500];
    gethostname(hostname, 500);
    
    cout <<"#"<<endl<<"# AmplitudeLib v. " << N.Version()  << " running on " << hostname << endl;
    cout <<"# Now is " << today ;
    cout <<"#"<<endl;
	delete[] hostname;

    cout << "# " << N.GetString() << endl;
    cout << "# Jet mass = " << mass << " GeV, low pt cutoff = " << LOW_PT_CUT << " GeV";
    cout << "# PDF: " << pdf->GetString() << endl;
    cout <<"# Rapidity shift: " << rapidity_shift << endl;
    
    // Print quarks
    std::vector<Parton> ps;
    ps.push_back(U);
	ps.push_back(UBAR);
	ps.push_back(D);
	ps.push_back(DBAR);
	ps.push_back(S);
	ps.push_back(SBAR);
    ps.push_back(C);
	ps.push_back(CBAR);
    ps.push_back(B);
    ps.push_back(BBAR);
	
    ps.push_back(G);
    xs.SetPartons(ps);
    cout <<"# Partons: " ;
    for (unsigned int i=0; i<xs.Partons().size(); i++)
    {
        cout << PartonToString(xs.Partons()[i]) << " ";
    }
    cout << endl;
    
    if (dps == true)
        cout << "# Computing DPS contribution" << endl;

    //////////////////////////////////////
    cout << "# minE   maxE   yield [note: for p, multiply by sigma0/2 / sigma_inel, for nuke: do b int]" << endl;
    
    for (double energy = minE; energy + binwidth <= maxE; energy += Estep)
    {
        inthelper_castor par;
        par.xs = &xs;
        par.minE=energy;
        par.maxE=energy + binwidth;
        par.sqrts=sqrts;
        par.pdf=pdf;
        par.m=mass;
        
         double result,abserr;
        
        
        if (dps == false)
        {
            gsl_function fun;
            fun.params=&par;
            fun.function = inthelperf_y;
            
            
            
            gsl_integration_workspace *workspace
            = gsl_integration_workspace_alloc(INTWORKSPACEDIVISIONS);
            gsl_integration_workspace *workspace_internal
            = gsl_integration_workspace_alloc(INTWORKSPACEDIVISIONS);
            par.workspace = workspace_internal;
            int status;
            double miny=castor_min_pseudorapidity-2; double maxy = castor_max_pseudorapidity+2;
            status = gsl_integration_qag(&fun, miny, maxy, 0, INTACCURACY, INTWORKSPACEDIVISIONS, GSL_INTEG_GAUSS51, workspace, &result, &abserr);
            if (status)
                cerr << "y integral failed at " << LINEINFO <<": result " << result
                    << " relerror " << std::abs(abserr/result) << endl;
            
            gsl_integration_workspace_free(workspace);
            gsl_integration_workspace_free(workspace_internal);
            
        }
        else
        {
            const gsl_rng_type * rngtype = gsl_rng_default;
            gsl_rng_env_setup();
            gsl_rng* rng = gsl_rng_alloc(rngtype);
            
            double *lower;
            double *upper;
            lower = new double[DPS_N*2];
            upper = new double[DPS_N*2];
            
            gsl_monte_function F;
            if (DPS_N==2)
            {
                F.f = inthelperf_dps_mc;
                F.dim=4;
                lower[0]=lower[1]=LOW_PT_CUT;
                lower[2]=lower[3]=castor_min_pseudorapidity - rapidity_shift;
                upper[0]=upper[1]=30;
                upper[2]=upper[3]=castor_max_pseudorapidity - rapidity_shift;

            }
            else if (DPS_N == 3)
            {
                F.f = inthelperf_3ps_mc;
                F.dim=6;
                lower[0]=lower[1]=lower[2]=LOW_PT_CUT;
                lower[3]=lower[4]=lower[5]=castor_min_pseudorapidity - rapidity_shift;
                upper[0]=upper[1]=upper[2]=30;
                upper[3]=upper[4]=upper[5]=castor_max_pseudorapidity - rapidity_shift;
            }
            F.params = &par;
            gsl_monte_miser_state *s = gsl_monte_miser_alloc(F.dim);
            
            // Vec is {pt1,pt2,y1,y2}
            
            cout << "# mcintpoints " << mcintpoints << endl;
            
            gsl_monte_miser_integrate(&F, lower, upper, F.dim, mcintpoints, rng, s, &result, &abserr);
            gsl_monte_miser_free(s);
            gsl_rng_free(rng);
            
            cout << "# MC integral uncertainty estimate " << 100.0*std::abs(abserr/result) << "%" << endl;
            
            
            delete[] lower;
            delete[] upper;
            
        }
    
    
    
    
        cout << energy << " " << energy+binwidth << " " << result << endl;
    }
    

    return 0;
}

double inthelperf_mc(double* vec, size_t dim, void* p)
{
    inthelper_castor *par = (inthelper_castor*)p;
    par->y = vec[1];
    return inthelperf_pt(vec[0], par);
}

double inthelperf_y(double y, void* p)
{
    //cout << "#New y integral " << y << endl;
    inthelper_castor *par = (inthelper_castor*)p;
    par->y=y;
    gsl_function fun;
    fun.params=par;
    fun.function = inthelperf_pt;
    
    // If you use this, then remember to free it
    //gsl_integration_workspace *workspace
    //= gsl_integration_workspace_alloc(INTWORKSPACEDIVISIONS);
    int status; double result,abserr;
    
    // Compute pt limits
    double shift = rapidity_shift;

	// Check that the parton ends up in Castor
    // Note that if we really sparate rapidity and pseudorapidity, this can
    // be done only in the next integrnad as we need also pt
    double y_lab = y + shift;
	//double y_lab = y + shift;
	//if (y_lab < castor_min_pseudorapidity or y_lab > castor_max_pseudorapidity)
	//	return 0;
    
    
    double minpt = std::sqrt( par->minE*par->minE - par->m*par->m/2.0 - 1.0/2.0*par->m*par->m*std::cosh(2.0*y_lab)) * 1.0 / std::cosh(y_lab) ;
    double maxpt =std::sqrt( par->maxE*par->maxE - par->m*par->m/2.0 - 1.0/2.0*par->m*par->m*std::cosh(2.0*y_lab)) * 1.0 / std::cosh(y_lab) ;
    
    
    if (minpt < 0.2 or maxpt < 0.2 or minpt > 100 or maxpt > 100 or isnan(minpt) or isnan(maxpt) or isinf(minpt) or isinf(maxpt))
        return 0;
    
    if (minpt < LOW_PT_CUT)
        return 0;
    
    status = gsl_integration_qag(&fun, minpt, maxpt, 0, INTACCURACY, INTWORKSPACEDIVISIONS, GSL_INTEG_GAUSS51, par->workspace, &result, &abserr);
    
    if (status)
        cerr << "pt integral failed, result " << result << " relerr " << std::abs(abserr/result) << " y " << y << endl;
    
    return result;
}

double inthelperf_pt(double pt, void* p)
{

    inthelper_castor *par = (inthelper_castor*)p;
    
    // Kinematics
    // The calculation is done in the center-of-mass frame
    // But castor measures jet energy in the LAB frame
    
    // First approximation: we set rapidity=pseudorapidity
    // and compute the produced jet energy in the LAB frame
    // after applyint rapidity_shift boost
    
    // This should be quite good approximation, at least with m=0.2GV,
    // as pt values are aloways > 1 GeV
    
    
    double shift = rapidity_shift;
 
    // Check kinematics
    double eta_lab = Pseudorapidity(par->y + shift, pt, par->m);
    if (eta_lab < castor_min_pseudorapidity or eta_lab > castor_max_pseudorapidity)
            return 0;
    
    double energy = JetEnergy(par->y + shift, pt, par->m);
    
    if (energy < par->minE or energy > par->maxE)
    {
        //cout << "Out of kinematics [" << par->minE << ", " << par->maxE << "]: y " << par->y << " pt " << pt << " E " << energy << endl;
        //cout << par->y << " " << pt <<" " << energy  << endl;
        return 0;
    }
    
    // Do angular integral, and add jacobian 2pi
    // 2nd to last parameters: false=no deuteron
    double pdf_scale = pt;
    if (pt < par->pdf->MinQ())
        pdf_scale = par->pdf->MinQ();
    double diffxs = par->xs->dHadronMultiplicity_dyd2pt_parton(par->y, pt, par->sqrts,
                                                               par->pdf, false,  pdf_scale );
    
	
    if (diffxs <0 )
    {
        cerr << "Differential cross section<0 at y " << par->y << " pt " << pt << " res " << diffxs << endl;
    }
    
    
    return 2.0*M_PI*pt*diffxs;;
}




double inthelperf_dps_mc(double* vec, size_t dim, void* p)
{
    
    inthelper_castor *par = (inthelper_castor*)p;
    
    double pt1 = vec[0];
    double pt2 = vec[1];
    double y1=vec[2];
    double y2=vec[3];
    
    // Kinematics
    // The calculation is done in the center-of-mass frame
    // But castor measures jet energy in the LAB frame
    
    // First approximation: we set rapidity=pseudorapidity
    // and compute the produced jet energy in the LAB frame
    // after applyint rapidity_shift boost
    
    // This should be quite good approximation, at least with m=0.2GV,
    // as pt values are aloways > 1 GeV
    
    
    double shift = rapidity_shift;
    
    double y1_lab = y1+shift;
    double y2_lab = y2+shift;
    
    double eta1_lab = Pseudorapidity(y1_lab, pt1, par->m);
    double eta2_lab = Pseudorapidity(y2_lab, pt2, par->m);
    
    // End up in CASTOR
    if (eta1_lab < castor_min_pseudorapidity or eta1_lab > castor_max_pseudorapidity or eta2_lab < castor_min_pseudorapidity or eta2_lab > castor_max_pseudorapidity)
        return 0;
    
    
    // Check kinematics
    double energy_1 = JetEnergy(y1_lab, pt1, par->m);
    double energy_2 = JetEnergy(y2_lab, pt2, par->m);
    double energy = energy_1 + energy_2;
    
    if (energy < par->minE or energy > par->maxE)
    {
        //cout << "Out of kinematics [" << par->minE << ", " << par->maxE << "]: y " << par->y << " pt " << pt << " E " << energy << endl;
        //cout << par->y << " " << pt <<" " << energy  << endl;
        return 0;
    }
    
    // Do angular integral, and add jacobian 2pi
    // 2nd to last parameters: false=no deuteron
    double pdf_scale = 0.5*(pt1+pt2); // negative = automatic
    if (pdf_scale < par->pdf->MinQ())
        pdf_scale = par->pdf->MinQ();
    double diffxs = par->xs->dHadronMultiplicity_dyd2pt_parton_dps(y1,pt1,y2,pt2, par->sqrts,
                                                               par->pdf, false,  pdf_scale );
    
    
    if (diffxs <0 )
    {
        cerr << "Differential cross section<0, res " << diffxs << endl;
    }
    
    // (2pi)^2 from two angular integrals, this should later be scaled by R/2pi to take into account
    //the jet finding algorithm effect
    return SQR(2.0*M_PI)*pt1*pt2*diffxs;
}




///////////////////////////////////////////
///////////////////// 3 particle production
///////////////////////////////////////////
double inthelperf_3ps_mc(double* vec, size_t dim, void* p)
{
    
    inthelper_castor *par = (inthelper_castor*)p;
    
    double pt1 = vec[0];
    double pt2 = vec[1];
    double pt3 = vec[2];
    double y1=vec[3];
    double y2=vec[4];
    double y3 = vec[5];
    
    // Kinematics
    // The calculation is done in the center-of-mass frame
    // But castor measures jet energy in the LAB frame
    
    // First approximation: we set rapidity=pseudorapidity
    // and compute the produced jet energy in the LAB frame
    // after applyint rapidity_shift boost
    
    // This should be quite good approximation, at least with m=0.2GV,
    // as pt values are aloways > 1 GeV
    
    
    double shift = rapidity_shift;
    
    double y1_lab = y1+shift;
    double y2_lab = y2+shift;
    double y3_lab = y3 + shift;
    double eta1_lab = Pseudorapidity(y1_lab, pt1, par->m);
    double eta2_lab = Pseudorapidity(y2_lab, pt2, par->m);
    double eta3_lab = Pseudorapidity(y3_lab, pt3, par->m);
    
    // End up in CASTOR
    if (eta1_lab < castor_min_pseudorapidity or eta1_lab > castor_max_pseudorapidity or eta2_lab < castor_min_pseudorapidity or eta2_lab > castor_max_pseudorapidity
        or eta3_lab < castor_min_pseudorapidity or eta3_lab > castor_max_pseudorapidity)
        return 0;
    
    
    // Check kinematics
    double energy_1 = JetEnergy(y1_lab, pt1, par->m);
    double energy_2 = JetEnergy(y2_lab, pt2, par->m);
    double energy_3 = JetEnergy(y3_lab, pt3, par->m);
    double energy = energy_1 + energy_2 + energy_3;
    
    if (energy < par->minE or energy > par->maxE)
    {
        //cout << "Out of kinematics [" << par->minE << ", " << par->maxE << "]: y " << par->y << " pt " << pt << " E " << energy << endl;
        //cout << par->y << " " << pt <<" " << energy  << endl;
        return 0;
    }
    
    // Do angular integral, and add jacobian 2pi
    // 2nd to last parameters: false=no deuteron
    double pdf_scale = (pt1+pt2+pt3)/3.0;
    if (pdf_scale < par->pdf->MinQ())
        pdf_scale = par->pdf->MinQ();
    double diffxs = par->xs->dHadronMultiplicity_dyd2pt_parton_3ps(y1,pt1,y2,pt2, y3, pt3, par->sqrts,
                                                                   par->pdf, false,  pdf_scale );
    
    
    if (diffxs <0 )
    {
        cerr << "Differential cross section<0, res " << diffxs << endl;
    }
    
    // (2pi)^3 from two angular integrals, this should later be scaled by R/2pi to take into account
    //the jet finding algorithm effect
    return std::pow(2.0*M_PI, 3.0)*pt1*pt2*pt3*diffxs;
}






