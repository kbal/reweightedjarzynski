/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "bias/ReweightBase.h"
#include "core/ActionRegister.h"
#include "tools/Matrix.h"

//+PLUMEDOC REWEIGHTING REWEIGHT_GEOMFES
/*
Calculate a gauge correction to a probability distribution. 
Can be used to calculate the geometric free energy surface.

The calculation of free energy barriers requires the use of the so-called 
geometric free energy surface \f$F^G (s)\f$, which is defined as:

\f[
F^G(s) = F(s) - \frac{1}{\beta} \ln \left \langle \lambda |\nabla s| \right \rangle_s
\f]

Therefore, the geometric FES can be calculated through \ref HISTOGRAM 
averaging in a biased simulaton using:

\f[
F^G(s) = -\frac{1}{\beta} \ln \langle w(t) \cdot \delta [s - s(t)] \cdot \lambda |\nabla s|  \rangle_b
\f]

Here, \f$w (t)\f$ is weight that removes the effect of a bias potential, 
such as \ref REWEIGHT_BIAS.

This code implements a generlized form of the above equations to deal 
with histograms involving multiple CVs.

\par Examples

In the following example we use the distance between atoms 1 and 2 as CV. 
We can now calculate both the geometric FES, as well as the standard FES. 
The only difference between the two FESes is that REWEIGHT_GEOMFES is only 
invoked in one \ref HISTOGRAM, trough the LOGWEIGHTS keyword.
The histograms are converted into a FES, and written to files.

\plumedfile
DISTANCE ATOMS=1,2 LABEL=x
REWEIGHT_GEOMFES ARG=x TEMP=300 LABEL=xgeom

HISTOGRAM ...
  ARG=x
  GRID_MIN=0.0
  GRID_MAX=3.0
  GRID_BIN=300
  BANDWIDTH=0.05
  LABEL=hstd
... HISTOGRAM

HISTOGRAM ...
  ARG=x
  GRID_MIN=0.0
  GRID_MAX=3.0
  GRID_BIN=300
  BANDWIDTH=0.05
  LOGWEIGHTS=xgeom
  LABEL=hstd
... HISTOGRAM

CONVERT_TO_FES GRID=hstd  TEMP=300 LABEL=stdfes
CONVERT_TO_FES GRID=hgeom TEMP=300 LABEL=geomfes

DUMPGRID GRID=stdfes  FILE=fes_std  STRIDE=1000000
DUMPGRID GRID=geomfes FILE=fes_geom STRIDE=1000000

\endplumedfile

A LOGWEIGHTS keyword can take multiple arguments, and can thus be combined with 
\ref REWEIGHT_BIAS or \ref REWEIGHT_METAD, which will provide the \f$w (t)\f$ 
mentioned earlier.


*/
//+ENDPLUMEDOC

namespace PLMD {
namespace bias {

class ReweightGeomFES : 
  public ReweightBase
{
public:
  static void registerKeywords(Keywords&);
  explicit ReweightGeomFES(const ActionOptions&ao);
  double getLogWeight() override;
  bool checkNeedsGradients()const override {return true;}
};

PLUMED_REGISTER_ACTION(ReweightGeomFES,"REWEIGHT_GEOMFES")

void ReweightGeomFES::registerKeywords(Keywords& keys ) {
  ReweightBase::registerKeywords( keys );
  keys.use("ARG");
}

ReweightGeomFES::ReweightGeomFES(const ActionOptions&ao):
  Action(ao),
  ReweightBase(ao)
{
  checkRead();
}

double ReweightGeomFES::getLogWeight() {
  const int ndim = getNumberOfArguments();
  Matrix<double> grads(ndim,ndim);
  double val;
  // Get the projected gradient
  for (int i = 0; i < ndim; ++i){
    for (int j = i; j < ndim; ++j){
      grads(i,j)=grads(j,i)=getProjection(i,j);
    }
  }
  // Calculate correction to FES as square root of determinant
  logdet( grads, val );
  return val/2.0;
}

}
}
