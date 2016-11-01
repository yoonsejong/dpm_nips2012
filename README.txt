This is a reference MATLAB implementation of Distirubted Probabilistic Principal Component Analysis (D-PPCA). The code is provided under GPLv2.

===
GPL
===

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 2 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

========
CITATION
========

If you used this program for any of your work, please cite the publication below:

S. Yoon and V. Pavlovic. "Distributed Probabilistic Learning for Camera Networks with Missing Data". In Advances in Neural Information Processing Systems 25 (NIPS) 2012.

===================
HOW TO RUN THE CODE
===================

Requires a reasonably recent version of MATLAB (2011-). Just hit 

>> demo_all

from command window. If you have Parallel Computing Toolbox, it's recommended to run 

>> matlabpool;

in advance for faster computation. For MATLAB 2013a and later, parpool will automatically launch if the toolbox is installed.

=======================
OTHER DATASET AND TOOLS
=======================

For Caltech Turntable dataset, please refer: http://www.vision.caltech.edu/Image_Datasets/3D_objects/

For Hopkins 155 dataset, please refer: http://www.vision.jhu.edu/data/hopkins155/

For Voodoo Camera Tracker, please see: http://www.viscoda.com/index.php/en/products/non-commercial/voodoo-camera-tracker
