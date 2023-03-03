/*
Copyright 2023 Joshuah Rainstar joshuah.rainstar@gmail.com
Permission is hereby granted, free of charge, to any person obtaining a copy of this softwareand associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and /or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*
Copyright 2023 Joshuah Rainstar
This program is free software; you can redistribute it and /or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110 - 1301, USA.
*/
//please note: this project is still a work in process and is not finished.
//please do not attempt to use this code for any purpose until this line is removed.


/*
* Cleanup. CPP version 1.10 2/25/23
* all work has been revised and is becoming close(r) to correct.
* convolution is now 100% guaranteed to behave like the python.
* entropy now behaves like the python.
* as far as I can tell, everything will and should act like the python *except* the inverse short time fourier transform(overlapping).
* as a result, we are now shipping this as 1.0 despite it not being identical or even similar.
*
*
*Instructions to future developers:
*I may not be alive forever. If I die, or stop developing on this work, anyone is free and welcome to fork it, extend it, enhance it, fix it.
* 
*/


//https://stackoverflow.com/questions/39675436/how-to-get-fftw-working-on-windows-for-dummies
//http://ftp.fftw.org/install/windows.html
//Usage Instructions:
//for visual studio 2017 onwards:
//generate the .lib for fftw3f.
//on the right, in visual studio, you'll notice a visual folder hierarchy for the solution.
//right click and add - the correct header to the headers, the .lib to the libraries.
//should just compile, then copy the fftw3f dll to the output folder.
//NOTE : If this is for a DLL, you must remove the .main function.
//Note: linux is not supported, any linux headers included are simply for linux developer's convenience.



#include <iostream>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <array>
#include <execution>
#include "fftw3.h"



#if defined(_MSC_VER)
	//  Microsoft 
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
	//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
	//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

//TODO: find ways to merge the uses of the above so that a mimimum in working memory can be utilized


//export our functions - this is only for use on linux.
#if defined(__GNUC__)
EXPORT void setConstant(float val);
EXPORT void set_NBINS(int val);
EXPORT std::array<float, 8192> process(std::array<float, 8192> input);
#endif

# define M_PI           3.14159265358979323846  /* pi */

class EXPORT Filter
{
private:


	//Filter Class Commentary

	//You just pass 8192 floats into the class, and get 8192 back. that's it.
	//it only works at a sample rate of 48000. Anything else and I can't guarantee anything.
	//you can pass a different entropy constant and NBINS. 
	//for more info see the class constructor at the head of the public section of the class.
	//using fftw, processes 2 seconds of audio in 33 ms.
	
	//note: the python shifts the windows. These windows have been pre ifftshifted for use in the stft class
	static constexpr std::array<float, 512> shifted_logistic_window{ 1., 0.94409333, 0.88818667, 0.85535081, 0.83195987, 0.81374363, 0.79880003, 0.78611443, 0.77508119, 0.76530966, 0.75653308, 0.7485612, 0.74125354, 0.73450341, 0.72822787, 0.72236118, 0.71685033, 0.71165198, 0.7067302, 0.70205491, 0.69760064, 0.69334567, 0.68927131, 0.68536134, 0.68160163, 0.67797977, 0.67448482, 0.67110707, 0.66783786, 0.66466946, 0.6615949, 0.65860793, 0.65570285, 0.65287452, 0.65011824, 0.64742972, 0.64480505, 0.64224061, 0.63973308, 0.63727941, 0.63487677, 0.63252254, 0.63021427, 0.62794972, 0.62572677, 0.62354345, 0.62139793, 0.61928848, 0.6172135, 0.61517145, 0.61316093, 0.61118059, 0.60922916, 0.60730545, 0.60540833, 0.60353673, 0.60168964, 0.59986611, 0.59806522, 0.5962861, 0.59452794, 0.59278996, 0.5910714, 0.58937155, 0.58768973, 0.5860253, 0.58437763, 0.58274613, 0.58113023, 0.57952939, 0.57794307, 0.57637077, 0.57481202, 0.57326635, 0.5717333, 0.57021245, 0.56870339, 0.56720571, 0.56571903, 0.56424297, 0.56277718, 0.56132129, 0.55987498, 0.55843793, 0.5570098, 0.55559029, 0.55417912, 0.55277598, 0.55138059, 0.54999269, 0.548612, 0.54723826, 0.54587123, 0.54451066, 0.5431563, 0.54180793, 0.54046531, 0.53912822, 0.53779645, 0.53646977, 0.53514798, 0.53383087, 0.53251824, 0.53120989, 0.52990563, 0.52860527, 0.52730862, 0.5260155, 0.52472572, 0.5234391, 0.52215548, 0.52087466, 0.51959649, 0.51832079, 0.51704739, 0.51577613, 0.51450685, 0.51323937, 0.51197354, 0.5107092, 0.50944618, 0.50818433, 0.5069235, 0.50566352, 0.50440424, 0.5031455, 0.50188714, 0.50062902, 0.49937098, 0.49811286, 0.4968545, 0.49559576, 0.49433648, 0.4930765, 0.49181567, 0.49055382, 0.4892908, 0.48802646, 0.48676063, 0.48549315, 0.48422387, 0.48295261, 0.48167921, 0.48040351, 0.47912534, 0.47784452, 0.4765609, 0.47527428, 0.4739845, 0.47269138, 0.47139473, 0.47009437, 0.46879011, 0.46748176, 0.46616913, 0.46485202, 0.46353023, 0.46220355, 0.46087178, 0.45953469, 0.45819207, 0.4568437, 0.45548934, 0.45412877, 0.45276174, 0.451388, 0.45000731, 0.44861941, 0.44722402, 0.44582088, 0.44440971, 0.4429902, 0.44156207, 0.44012502, 0.43867871, 0.43722282, 0.43575703, 0.43428097, 0.43279429, 0.43129661, 0.42978755, 0.4282667, 0.42673365, 0.42518798, 0.42362923, 0.42205693, 0.42047061, 0.41886977, 0.41725387, 0.41562237, 0.4139747, 0.41231027, 0.41062845, 0.4089286, 0.40721004, 0.40547206, 0.4037139, 0.40193478, 0.40013389, 0.39831036, 0.39646327, 0.39459167, 0.39269455, 0.39077084, 0.38881941, 0.38683907, 0.38482855, 0.3827865, 0.38071152, 0.37860207, 0.37645655, 0.37427323, 0.37205028, 0.36978573, 0.36747746, 0.36512323, 0.36272059, 0.36026692, 0.35775939, 0.35519495, 0.35257028, 0.34988176, 0.34712548, 0.34429715, 0.34139207, 0.3384051, 0.33533054, 0.33216214, 0.32889293, 0.32551518, 0.32202023, 0.31839837, 0.31463866, 0.31072869, 0.30665433, 0.30239936, 0.29794509, 0.2932698, 0.28834802, 0.28314967, 0.27763882, 0.27177213, 0.26549659, 0.25874646, 0.2514388, 0.24346692, 0.23469034, 0.22491881, 0.21388557, 0.20119997, 0.18625637, 0.16804013, 0.14464919, 0.11181333, 0.05590667, 0., 0., 0.05590667, 0.11181333, 0.14464919, 0.16804013, 0.18625637, 0.20119997, 0.21388557, 0.22491881, 0.23469034, 0.24346692, 0.2514388, 0.25874646, 0.26549659, 0.27177213, 0.27763882, 0.28314967, 0.28834802, 0.2932698, 0.29794509, 0.30239936, 0.30665433, 0.31072869, 0.31463866, 0.31839837, 0.32202023, 0.32551518, 0.32889293, 0.33216214, 0.33533054, 0.3384051, 0.34139207, 0.34429715, 0.34712548, 0.34988176, 0.35257028, 0.35519495, 0.35775939, 0.36026692, 0.36272059, 0.36512323, 0.36747746, 0.36978573, 0.37205028, 0.37427323, 0.37645655, 0.37860207, 0.38071152, 0.3827865, 0.38482855, 0.38683907, 0.38881941, 0.39077084, 0.39269455, 0.39459167, 0.39646327, 0.39831036, 0.40013389, 0.40193478, 0.4037139, 0.40547206, 0.40721004, 0.4089286, 0.41062845, 0.41231027, 0.4139747, 0.41562237, 0.41725387, 0.41886977, 0.42047061, 0.42205693, 0.42362923, 0.42518798, 0.42673365, 0.4282667, 0.42978755, 0.43129661, 0.43279429, 0.43428097, 0.43575703, 0.43722282, 0.43867871, 0.44012502, 0.44156207, 0.4429902, 0.44440971, 0.44582088, 0.44722402, 0.44861941, 0.45000731, 0.451388, 0.45276174, 0.45412877, 0.45548934, 0.4568437, 0.45819207, 0.45953469, 0.46087178, 0.46220355, 0.46353023, 0.46485202, 0.46616913, 0.46748176, 0.46879011, 0.47009437, 0.47139473, 0.47269138, 0.4739845, 0.47527428, 0.4765609, 0.47784452, 0.47912534, 0.48040351, 0.48167921, 0.48295261, 0.48422387, 0.48549315, 0.48676063, 0.48802646, 0.4892908, 0.49055382, 0.49181567, 0.4930765, 0.49433648, 0.49559576, 0.4968545, 0.49811286, 0.49937098, 0.50062902, 0.50188714, 0.5031455, 0.50440424, 0.50566352, 0.5069235, 0.50818433, 0.50944618, 0.5107092, 0.51197354, 0.51323937, 0.51450685, 0.51577613, 0.51704739, 0.51832079, 0.51959649, 0.52087466, 0.52215548, 0.5234391, 0.52472572, 0.5260155, 0.52730862, 0.52860527, 0.52990563, 0.53120989, 0.53251824, 0.53383087, 0.53514798, 0.53646977, 0.53779645, 0.53912822, 0.54046531, 0.54180793, 0.5431563, 0.54451066, 0.54587123, 0.54723826, 0.548612, 0.54999269, 0.55138059, 0.55277598, 0.55417912, 0.55559029, 0.5570098, 0.55843793, 0.55987498, 0.56132129, 0.56277718, 0.56424297, 0.56571903, 0.56720571, 0.56870339, 0.57021245, 0.5717333, 0.57326635, 0.57481202, 0.57637077, 0.57794307, 0.57952939, 0.58113023, 0.58274613, 0.58437763, 0.5860253, 0.58768973, 0.58937155, 0.5910714, 0.59278996, 0.59452794, 0.5962861, 0.59806522, 0.59986611, 0.60168964, 0.60353673, 0.60540833, 0.60730545, 0.60922916, 0.61118059, 0.61316093, 0.61517145, 0.6172135, 0.61928848, 0.62139793, 0.62354345, 0.62572677, 0.62794972, 0.63021427, 0.63252254, 0.63487677, 0.63727941, 0.63973308, 0.64224061, 0.64480505, 0.64742972, 0.65011824, 0.65287452, 0.65570285, 0.65860793, 0.6615949, 0.66466946, 0.66783786, 0.67110707, 0.67448482, 0.67797977, 0.68160163, 0.68536134, 0.68927131, 0.69334567, 0.69760064, 0.70205491, 0.7067302, 0.71165198, 0.71685033, 0.72236118, 0.72822787, 0.73450341, 0.74125354, 0.7485612, 0.75653308, 0.76530966, 0.77508119, 0.78611443, 0.79880003, 0.81374363, 0.83195987, 0.85535081, 0.88818667, 0.94409333, 1. };
	static constexpr std::array<float, 512> shifted_hann_window = { 9.99990551e-01, 9.99914959e-01, 9.99763787e-01, 9.99537058e-01, 9.99234805e-01, 9.98857075e-01, 9.98403924e-01, 9.97875422e-01, 9.97271648e-01, 9.96592693e-01, 9.95838660e-01, 9.95009663e-01, 9.94105827e-01, 9.93127290e-01, 9.92074198e-01, 9.90946711e-01, 9.89745000e-01, 9.88469246e-01, 9.87119643e-01, 9.85696393e-01, 9.84199713e-01, 9.82629829e-01, 9.80986977e-01, 9.79271407e-01, 9.77483377e-01, 9.75623159e-01, 9.73691033e-01, 9.71687291e-01, 9.69612237e-01, 9.67466184e-01, 9.65249456e-01, 9.62962389e-01, 9.60605328e-01, 9.58178630e-01, 9.55682662e-01, 9.53117800e-01, 9.50484434e-01, 9.47782960e-01, 9.45013788e-01, 9.42177336e-01, 9.39274033e-01, 9.36304317e-01, 9.33268638e-01, 9.30167455e-01, 9.27001237e-01, 9.23770461e-01, 9.20475618e-01, 9.17117204e-01, 9.13695728e-01, 9.10211707e-01, 9.06665667e-01, 9.03058145e-01, 8.99389686e-01, 8.95660845e-01, 8.91872186e-01, 8.88024281e-01, 8.84117711e-01, 8.80153069e-01, 8.76130952e-01, 8.72051970e-01, 8.67916738e-01, 8.63725883e-01, 8.59480037e-01, 8.55179843e-01, 8.50825950e-01, 8.46419017e-01, 8.41959711e-01, 8.37448705e-01, 8.32886681e-01, 8.28274329e-01, 8.23612347e-01, 8.18901439e-01, 8.14142317e-01, 8.09335702e-01, 8.04482319e-01, 7.99582902e-01, 7.94638193e-01, 7.89648938e-01, 7.84615893e-01, 7.79539817e-01, 7.74421479e-01, 7.69261652e-01, 7.64061117e-01, 7.58820659e-01, 7.53541070e-01, 7.48223150e-01, 7.42867702e-01, 7.37475536e-01, 7.32047467e-01, 7.26584315e-01, 7.21086907e-01, 7.15556073e-01, 7.09992651e-01, 7.04397480e-01, 6.98771407e-01, 6.93115283e-01, 6.87429962e-01, 6.81716305e-01, 6.75975174e-01, 6.70207439e-01, 6.64413970e-01, 6.58595644e-01, 6.52753341e-01, 6.46887944e-01, 6.41000339e-01, 6.35091417e-01, 6.29162070e-01, 6.23213197e-01, 6.17245695e-01, 6.11260467e-01, 6.05258418e-01, 5.99240456e-01, 5.93207489e-01, 5.87160431e-01, 5.81100196e-01, 5.75027699e-01, 5.68943859e-01, 5.62849596e-01, 5.56745831e-01, 5.50633486e-01, 5.44513487e-01, 5.38386758e-01, 5.32254225e-01, 5.26116815e-01, 5.19975458e-01, 5.13831080e-01, 5.07684611e-01, 5.01536980e-01, 4.95389117e-01, 4.89241951e-01, 4.83096412e-01, 4.76953428e-01, 4.70813928e-01, 4.64678841e-01, 4.58549094e-01, 4.52425614e-01, 4.46309327e-01, 4.40201156e-01, 4.34102027e-01, 4.28012860e-01, 4.21934577e-01, 4.15868096e-01, 4.09814335e-01, 4.03774209e-01, 3.97748631e-01, 3.91738511e-01, 3.85744760e-01, 3.79768282e-01, 3.73809982e-01, 3.67870760e-01, 3.61951513e-01, 3.56053138e-01, 3.50176526e-01, 3.44322565e-01, 3.38492141e-01, 3.32686134e-01, 3.26905422e-01, 3.21150881e-01, 3.15423378e-01, 3.09723782e-01, 3.04052952e-01, 2.98411747e-01, 2.92801019e-01, 2.87221617e-01, 2.81674384e-01, 2.76160159e-01, 2.70679775e-01, 2.65234062e-01, 2.59823842e-01, 2.54449933e-01, 2.49113148e-01, 2.43814294e-01, 2.38554171e-01, 2.33333576e-01, 2.28153297e-01, 2.23014117e-01, 2.17916814e-01, 2.12862158e-01, 2.07850913e-01, 2.02883837e-01, 1.97961681e-01, 1.93085190e-01, 1.88255099e-01, 1.83472140e-01, 1.78737036e-01, 1.74050502e-01, 1.69413247e-01, 1.64825973e-01, 1.60289372e-01, 1.55804131e-01, 1.51370928e-01, 1.46990432e-01, 1.42663307e-01, 1.38390206e-01, 1.34171776e-01, 1.30008654e-01, 1.25901469e-01, 1.21850843e-01, 1.17857388e-01, 1.13921708e-01, 1.10044397e-01, 1.06226043e-01, 1.02467221e-01, 9.87685015e-02, 9.51304424e-02, 9.15535940e-02, 8.80384971e-02, 8.45856832e-02, 8.11956742e-02, 7.78689827e-02, 7.46061116e-02, 7.14075543e-02, 6.82737943e-02, 6.52053053e-02, 6.22025514e-02, 5.92659864e-02, 5.63960544e-02, 5.35931893e-02, 5.08578147e-02, 4.81903443e-02, 4.55911813e-02, 4.30607187e-02, 4.05993391e-02, 3.82074146e-02, 3.58853068e-02, 3.36333667e-02, 3.14519350e-02, 2.93413412e-02, 2.73019047e-02, 2.53339336e-02, 2.34377255e-02, 2.16135671e-02, 1.98617342e-02, 1.81824916e-02, 1.65760932e-02, 1.50427819e-02, 1.35827895e-02, 1.21963367e-02, 1.08836332e-02, 9.64487731e-03, 8.48025644e-03, 7.38994662e-03, 6.37411270e-03, 5.43290826e-03, 4.56647559e-03, 3.77494569e-03, 3.05843822e-03, 2.41706151e-03, 1.85091253e-03, 1.36007687e-03, 9.44628746e-04, 6.04630957e-04, 3.40134910e-04, 1.51180595e-04, 3.77965773e-05, 0.00000000e+00, 0.00000000e+00, 3.77965773e-05, 1.51180595e-04, 3.40134910e-04, 6.04630957e-04, 9.44628746e-04, 1.36007687e-03, 1.85091253e-03, 2.41706151e-03, 3.05843822e-03, 3.77494569e-03, 4.56647559e-03, 5.43290826e-03, 6.37411270e-03, 7.38994662e-03, 8.48025644e-03, 9.64487731e-03, 1.08836332e-02, 1.21963367e-02, 1.35827895e-02, 1.50427819e-02, 1.65760932e-02, 1.81824916e-02, 1.98617342e-02, 2.16135671e-02, 2.34377255e-02, 2.53339336e-02, 2.73019047e-02, 2.93413412e-02, 3.14519350e-02, 3.36333667e-02, 3.58853068e-02, 3.82074146e-02, 4.05993391e-02, 4.30607187e-02, 4.55911813e-02, 4.81903443e-02, 5.08578147e-02, 5.35931893e-02, 5.63960544e-02, 5.92659864e-02, 6.22025514e-02, 6.52053053e-02, 6.82737943e-02, 7.14075543e-02, 7.46061116e-02, 7.78689827e-02, 8.11956742e-02, 8.45856832e-02, 8.80384971e-02, 9.15535940e-02, 9.51304424e-02, 9.87685015e-02, 1.02467221e-01, 1.06226043e-01, 1.10044397e-01, 1.13921708e-01, 1.17857388e-01, 1.21850843e-01, 1.25901469e-01, 1.30008654e-01, 1.34171776e-01, 1.38390206e-01, 1.42663307e-01, 1.46990432e-01, 1.51370928e-01, 1.55804131e-01, 1.60289372e-01, 1.64825973e-01, 1.69413247e-01, 1.74050502e-01, 1.78737036e-01, 1.83472140e-01, 1.88255099e-01, 1.93085190e-01, 1.97961681e-01, 2.02883837e-01, 2.07850913e-01, 2.12862158e-01, 2.17916814e-01, 2.23014117e-01, 2.28153297e-01, 2.33333576e-01, 2.38554171e-01, 2.43814294e-01, 2.49113148e-01, 2.54449933e-01, 2.59823842e-01, 2.65234062e-01, 2.70679775e-01, 2.76160159e-01, 2.81674384e-01, 2.87221617e-01, 2.92801019e-01, 2.98411747e-01, 3.04052952e-01, 3.09723782e-01, 3.15423378e-01, 3.21150881e-01, 3.26905422e-01, 3.32686134e-01, 3.38492141e-01, 3.44322565e-01, 3.50176526e-01, 3.56053138e-01, 3.61951513e-01, 3.67870760e-01, 3.73809982e-01, 3.79768282e-01, 3.85744760e-01, 3.91738511e-01, 3.97748631e-01, 4.03774209e-01, 4.09814335e-01, 4.15868096e-01, 4.21934577e-01, 4.28012860e-01, 4.34102027e-01, 4.40201156e-01, 4.46309327e-01, 4.52425614e-01, 4.58549094e-01, 4.64678841e-01, 4.70813928e-01, 4.76953428e-01, 4.83096412e-01, 4.89241951e-01, 4.95389117e-01, 5.01536980e-01, 5.07684611e-01, 5.13831080e-01, 5.19975458e-01, 5.26116815e-01, 5.32254225e-01, 5.38386758e-01, 5.44513487e-01, 5.50633486e-01, 5.56745831e-01, 5.62849596e-01, 5.68943859e-01, 5.75027699e-01, 5.81100196e-01, 5.87160431e-01, 5.93207489e-01, 5.99240456e-01, 6.05258418e-01, 6.11260467e-01, 6.17245695e-01, 6.23213197e-01, 6.29162070e-01, 6.35091417e-01, 6.41000339e-01, 6.46887944e-01, 6.52753341e-01, 6.58595644e-01, 6.64413970e-01, 6.70207439e-01, 6.75975174e-01, 6.81716305e-01, 6.87429962e-01, 6.93115283e-01, 6.98771407e-01, 7.04397480e-01, 7.09992651e-01, 7.15556073e-01, 7.21086907e-01, 7.26584315e-01, 7.32047467e-01, 7.37475536e-01, 7.42867702e-01, 7.48223150e-01, 7.53541070e-01, 7.58820659e-01, 7.64061117e-01, 7.69261652e-01, 7.74421479e-01, 7.79539817e-01, 7.84615893e-01, 7.89648938e-01, 7.94638193e-01, 7.99582902e-01, 8.04482319e-01, 8.09335702e-01, 8.14142317e-01, 8.18901439e-01, 8.23612347e-01, 8.28274329e-01, 8.32886681e-01, 8.37448705e-01, 8.41959711e-01, 8.46419017e-01, 8.50825950e-01, 8.55179843e-01, 8.59480037e-01, 8.63725883e-01, 8.67916738e-01, 8.72051970e-01, 8.76130952e-01, 8.80153069e-01, 8.84117711e-01, 8.88024281e-01, 8.91872186e-01, 8.95660845e-01, 8.99389686e-01, 9.03058145e-01, 9.06665667e-01, 9.10211707e-01, 9.13695728e-01, 9.17117204e-01, 9.20475618e-01, 9.23770461e-01, 9.27001237e-01, 9.30167455e-01, 9.33268638e-01, 9.36304317e-01, 9.39274033e-01, 9.42177336e-01, 9.45013788e-01, 9.47782960e-01, 9.50484434e-01, 9.53117800e-01, 9.55682662e-01, 9.58178630e-01, 9.60605328e-01, 9.62962389e-01, 9.65249456e-01, 9.67466184e-01, 9.69612237e-01, 9.71687291e-01, 9.73691033e-01, 9.75623159e-01, 9.77483377e-01, 9.79271407e-01, 9.80986977e-01, 9.82629829e-01, 9.84199713e-01, 9.85696393e-01, 9.87119643e-01, 9.88469246e-01, 9.89745000e-01, 9.90946711e-01, 9.92074198e-01, 9.93127290e-01, 9.94105827e-01, 9.95009663e-01, 9.95838660e-01, 9.96592693e-01, 9.97271648e-01, 9.97875422e-01, 9.98403924e-01, 9.98857075e-01, 9.99234805e-01, 9.99537058e-01, 9.99763787e-01, 9.99914959e-01, 9.99990551e-01 };
	static constexpr std::array<float, 512> synthesis_window = { 0.00000000e+00,2.52493737e-05,1.00993617e-04,2.27221124e-04,4.03912573e-04,6.31040943e-04,9.08571512e-04,1.23646188e-03,1.61466197e-03,2.04311406e-03,2.52175277e-03,3.05050510e-03,3.62929044e-03,4.25802059e-03,4.93659976e-03,5.66492464e-03,6.44288435e-03,7.27036053e-03,8.14722732e-03,9.07335139e-03,1.00485920e-02,1.10728009e-02,1.21458227e-02,1.32674943e-02,1.44376456e-02,1.56560990e-02,1.69226697e-02,1.82371657e-02,1.95993878e-02,2.10091294e-02,2.24661772e-02,2.39703105e-02,2.55213015e-02,2.71189155e-02,2.87629109e-02,3.04530390e-02,3.21890442e-02,3.39706641e-02,3.57976294e-02,3.76696640e-02,3.95864853e-02,4.15478036e-02,4.35533228e-02,4.56027403e-02,4.76957466e-02,4.98320260e-02,5.20112561e-02,5.42331081e-02,5.64972471e-02,5.88033315e-02,6.11510136e-02,6.35399396e-02,6.59697493e-02,6.84400765e-02,7.09505488e-02,7.35007880e-02,7.60904099e-02,7.87190242e-02,8.13862349e-02,8.40916401e-02,8.68348324e-02,8.96153985e-02,9.24329196e-02,9.52869711e-02,9.81771232e-02,1.01102941e-01,1.04063982e-01,1.07059803e-01,1.10089950e-01,1.13153968e-01,1.16251394e-01,1.19381763e-01,1.22544602e-01,1.25739435e-01,1.28965780e-01,1.32223151e-01,1.35511057e-01,1.38829001e-01,1.42176485e-01,1.45553002e-01,1.48958043e-01,1.52391095e-01,1.55851640e-01,1.59339154e-01,1.62853113e-01,1.66392985e-01,1.69958235e-01,1.73548325e-01,1.77162713e-01,1.80800852e-01,1.84462192e-01,1.88146180e-01,1.91852259e-01,1.95579867e-01,1.99328441e-01,2.03097413e-01,2.06886212e-01,2.10694266e-01,2.14520996e-01,2.18365823e-01,2.22228164e-01,2.26107433e-01,2.30003042e-01,2.33914400e-01,2.37840913e-01,2.41781985e-01,2.45737016e-01,2.49705407e-01,2.53686555e-01,2.57679853e-01,2.61684694e-01,2.65700470e-01,2.69726568e-01,2.73762377e-01,2.77807281e-01,2.81860664e-01,2.85921908e-01,2.89990394e-01,2.94065503e-01,2.98146611e-01,3.02233096e-01,3.06324335e-01,3.10419702e-01,3.14518572e-01,3.18620319e-01,3.22724316e-01,3.26829934e-01,3.30936547e-01,3.35043526e-01,3.39150246e-01,3.43256086e-01,3.47360427e-01,3.51462648e-01,3.55562129e-01,3.59658251e-01,3.63750398e-01,3.67837950e-01,3.71920292e-01,3.75996808e-01,3.80066883e-01,3.84129905e-01,3.88185260e-01,3.92232339e-01,3.96270531e-01,4.00299230e-01,4.04317828e-01,4.08325721e-01,4.12322306e-01,4.16306982e-01,4.20279150e-01,4.24238213e-01,4.28183575e-01,4.32114645e-01,4.36030831e-01,4.39931546e-01,4.43816203e-01,4.47684220e-01,4.51535015e-01,4.55368011e-01,4.59182632e-01,4.62978306e-01,4.66754464e-01,4.70510539e-01,4.74245967e-01,4.77960189e-01,4.81652647e-01,4.85322788e-01,4.88970060e-01,4.92593918e-01,4.96193817e-01,4.99769218e-01,5.03319584e-01,5.06844384e-01,5.10343088e-01,5.13815172e-01,5.17260116e-01,5.20677401e-01,5.24066516e-01,5.27426952e-01,5.30758205e-01,5.34059774e-01,5.37331165e-01,5.40571886e-01,5.43781450e-01,5.46959376e-01,5.50105185e-01,5.53218405e-01,5.56298569e-01,5.59345212e-01,5.62357878e-01,5.65336111e-01,5.68279465e-01,5.71187495e-01,5.74059764e-01,5.76895840e-01,5.79695293e-01,5.82457703e-01,5.85182652e-01,5.87869728e-01,5.90518527e-01,5.93128647e-01,5.95699693e-01,5.98231278e-01,6.00723016e-01,6.03174532e-01,6.05585452e-01,6.07955411e-01,6.10284050e-01,6.12571014e-01,6.14815956e-01,6.17018534e-01,6.19178413e-01,6.21295262e-01,6.23368760e-01,6.25398589e-01,6.27384439e-01,6.29326006e-01,6.31222993e-01,6.33075109e-01,6.34882068e-01,6.36643595e-01,6.38359416e-01,6.40029269e-01,6.41652895e-01,6.43230043e-01,6.44760469e-01,6.46243937e-01,6.47680215e-01,6.49069080e-01,6.50410317e-01,6.51703715e-01,6.52949072e-01,6.54146194e-01,6.55294893e-01,6.56394987e-01,6.57446303e-01,6.58448674e-01,6.59401943e-01,6.60305956e-01,6.61160570e-01,6.61965647e-01,6.62721059e-01,6.63426683e-01,6.64082404e-01,6.64688115e-01,6.65243717e-01,6.65749117e-01,6.66204231e-01,6.66608983e-01,6.66963302e-01,6.67267128e-01,6.67520406e-01,6.67723090e-01,6.67875141e-01,6.67976529e-01,6.68027230e-01,6.68027230e-01,6.67976529e-01,6.67875141e-01,6.67723090e-01,6.67520406e-01,6.67267128e-01,6.66963302e-01,6.66608983e-01,6.66204231e-01,6.65749117e-01,6.65243717e-01,6.64688115e-01,6.64082404e-01,6.63426683e-01,6.62721059e-01,6.61965647e-01,6.61160570e-01,6.60305956e-01,6.59401943e-01,6.58448674e-01,6.57446303e-01,6.56394987e-01,6.55294893e-01,6.54146194e-01,6.52949072e-01,6.51703715e-01,6.50410317e-01,6.49069080e-01,6.47680215e-01,6.46243937e-01,6.44760469e-01,6.43230043e-01,6.41652895e-01,6.40029269e-01,6.38359416e-01,6.36643595e-01,6.34882068e-01,6.33075109e-01,6.31222993e-01,6.29326006e-01,6.27384439e-01,6.25398589e-01,6.23368760e-01,6.21295262e-01,6.19178413e-01,6.17018534e-01,6.14815956e-01,6.12571014e-01,6.10284050e-01,6.07955411e-01,6.05585452e-01,6.03174532e-01,6.00723016e-01,5.98231278e-01,5.95699693e-01,5.93128647e-01,5.90518527e-01,5.87869728e-01,5.85182652e-01,5.82457703e-01,5.79695293e-01,5.76895840e-01,5.74059764e-01,5.71187495e-01,5.68279465e-01,5.65336111e-01,5.62357878e-01,5.59345212e-01,5.56298569e-01,5.53218405e-01,5.50105185e-01,5.46959376e-01,5.43781450e-01,5.40571886e-01,5.37331165e-01,5.34059774e-01,5.30758205e-01,5.27426952e-01,5.24066516e-01,5.20677401e-01,5.17260116e-01,5.13815172e-01,5.10343088e-01,5.06844384e-01,5.03319584e-01,4.99769218e-01,4.96193817e-01,4.92593918e-01,4.88970060e-01,4.85322788e-01,4.81652647e-01,4.77960189e-01,4.74245967e-01,4.70510539e-01,4.66754464e-01,4.62978306e-01,4.59182632e-01,4.55368011e-01,4.51535015e-01,4.47684220e-01,4.43816203e-01,4.39931546e-01,4.36030831e-01,4.32114645e-01,4.28183575e-01,4.24238213e-01,4.20279150e-01,4.16306982e-01,4.12322306e-01,4.08325721e-01,4.04317828e-01,4.00299230e-01,3.96270531e-01,3.92232339e-01,3.88185260e-01,3.84129905e-01,3.80066883e-01,3.75996808e-01,3.71920292e-01,3.67837950e-01,3.63750398e-01,3.59658251e-01,3.55562129e-01,3.51462648e-01,3.47360427e-01,3.43256086e-01,3.39150246e-01,3.35043526e-01,3.30936547e-01,3.26829934e-01,3.22724316e-01,3.18620319e-01,3.14518572e-01,3.10419702e-01,3.06324335e-01,3.02233096e-01,2.98146611e-01,2.94065503e-01,2.89990394e-01,2.85921908e-01,2.81860664e-01,2.77807281e-01,2.73762377e-01,2.69726568e-01,2.65700470e-01,2.61684694e-01,2.57679853e-01,2.53686555e-01,2.49705407e-01,2.45737016e-01,2.41781985e-01,2.37840913e-01,2.33914400e-01,2.30003042e-01,2.26107433e-01,2.22228164e-01,2.18365823e-01,2.14520996e-01,2.10694266e-01,2.06886212e-01,2.03097413e-01,1.99328441e-01,1.95579867e-01,1.91852259e-01,1.88146180e-01,1.84462192e-01,1.80800852e-01,1.77162713e-01,1.73548325e-01,1.69958235e-01,1.66392985e-01,1.62853113e-01,1.59339154e-01,1.55851640e-01,1.52391095e-01,1.48958043e-01,1.45553002e-01,1.42176485e-01,1.38829001e-01,1.35511057e-01,1.32223151e-01,1.28965780e-01,1.25739435e-01,1.22544602e-01,1.19381763e-01,1.16251394e-01,1.13153968e-01,1.10089950e-01,1.07059803e-01,1.04063982e-01,1.01102941e-01,9.81771232e-02,9.52869711e-02,9.24329196e-02,8.96153985e-02,8.68348324e-02,8.40916401e-02,8.13862349e-02,7.87190242e-02,7.60904099e-02,7.35007880e-02,7.09505488e-02,6.84400765e-02,6.59697493e-02,6.35399396e-02,6.11510136e-02,5.88033315e-02,5.64972471e-02,5.42331081e-02,5.20112561e-02,4.98320260e-02,4.76957466e-02,4.56027403e-02,4.35533228e-02,4.15478036e-02,3.95864853e-02,3.76696640e-02,3.57976294e-02,3.39706641e-02,3.21890442e-02,3.04530390e-02,2.87629109e-02,2.71189155e-02,2.55213015e-02,2.39703105e-02,2.24661772e-02,2.10091294e-02,1.95993878e-02,1.82371657e-02,1.69226697e-02,1.56560990e-02,1.44376456e-02,1.32674943e-02,1.21458227e-02,1.10728009e-02,1.00485920e-02,9.07335139e-03,8.14722732e-03,7.27036053e-03,6.44288435e-03,5.66492464e-03,4.93659976e-03,4.25802059e-03,3.62929044e-03,3.05050510e-03,2.52175277e-03,2.04311406e-03,1.61466197e-03,1.23646188e-03,9.08571512e-04,6.31040943e-04,4.03912573e-04,2.27221124e-04,1.00993617e-04,2.52493737e-05,0.00000000e+00 };
	static constexpr std::array<float, 37> logit_37 = { 0.0, 0.08441118, 0.16882236, 0.2197072, 0.25693163, 0.28672628, 0.31187091, 0.33385255, 0.35356306, 0.37158192, 0.38830914, 0.40403462, 0.41897721, 0.43330834, 0.44716693, 0.46066936, 0.47391649, 0.4869987, 0.5, 0.5130013, 0.52608351, 0.53933064, 0.55283307, 0.56669166, 0.58102279, 0.59596538, 0.61169086, 0.62841808, 0.64643694, 0.66614745, 0.68812909, 0.71327372, 0.74306837, 0.7802928, 0.83117764, 0.91558882, 1.0 };

	//working memory
	std::array<float, 24576> audio = {};
	std::array<float, 25087> audio_padded = {};
	std::array<float, 384> buffer = {};
	std::array<float, 192> entropy_unmasked = {};
	std::array<float, 192> entropy_smoothed = {}; //192 + 12
	std::array<int, 192> entropy_thresholded = {};
	std::array<float, 257> logit_distribution = {};


	std::array<float, 8192> output = { 0 };
	std::array<float, 8192> empty = { 0 };//just a bunch of zeros


	float t = 0, initial = 0, multiplier = 0, MAXIMUM = 0.612217f, constant_temp = 0, test = 0, thresh1 = 0.0f;
	int flag = 0, count = 0, sample_rate = 48000, N_FFT = 512, hop_size = 128;
	float CONST_1 = 0.057f, CONST_last = 0.057f;
	int NBINS_1 = 37, NBINS_last = 0;
	static constexpr int TIME_PAD = 13;
	static constexpr int FREQ_PAD = 3;

	std::array<std::complex<float>, 512> temp_complex_512 = {};
	std::array<float, 512> temp_512 = {};

	std::array<std::complex<float>, 257> temp_complex = {};
	std::array<float, 257> temp_257 = {};
	std::array<float, 128> temp_128 = {};


	std::array<std::array<std::complex<float>, 192>, 257> stft_complex = {};
	std::array<std::array<std::complex<float>, 64>, 257> stft_output = {};
	std::array<std::array<std::complex<float>, 64>, 257> stft_zeros = {}; // for use in the scenario two where we're processing the residual buffer but without the audio
	std::array<std::array<float, 192>, 257> stft_real = {};
	std::array<std::array<float, 192>, 257> smoothed = {};
	std::array<std::array<float, 192>, 257> previous = {};
	std::array<std::array<float, 26 + 192>, 6 + 257 > vertical = {};
	std::array<std::array<float, 26 + 192 >, 6 + 257> horizontal = {};
	//note this is not completely equivalent to same mode convolve. we pad extra and we conserve the products of the smoothing outwards until the end.
	std::array<float, 6 + 257> frequencywise_storage = {};
	/// <summary>
	/// name is a misnomer, it finds the ATD + the man.
	/// The MAN is an attempt to improve over MAD, ATD in like measure.
	/// Both empirically determined by someone with high school level algebra retention.
	/// </summary>
	/// <param name="data"></param>
	/// <returns></returns>
	inline float MAN(std::array<float, 257>& data) {
		std::array<float, 257> arr;
		int n = 0;

		// Remove NaN values from arr
		for (int i = 0; i < NBINS_last; i++) {
			if (!std::isnan(data[i])) {
				arr[n] = data[i];
				n++;
			}
		}

		// If v is empty, return 0.0 early
		if (n == 0) {
			return 0.0f;
		}

		// Compute the median of arr
		std::sort(arr.begin(), arr.begin() + n);
		float t = (n % 2 == 0) ? (arr[n / 2] + arr[(n / 2) - 1]) / 2.0f : arr[n / 2];

		// Compute the median of the absolute difference between arr and t
		std::array<float, 257> non_zero_diff;
		for (int i = 0; i < n; i++) {
			non_zero_diff[i] = std::abs(arr[i] - t);
		}
		std::sort(non_zero_diff.begin(), non_zero_diff.begin() + n);
		float e = (n % 2 == 0) ? (non_zero_diff[n / 2] + non_zero_diff[(n / 2) - 1]) / 2.0f : non_zero_diff[n / 2];

		// Compute the square root of the mean of the squared absolute deviation from e
		float sum_x = 0.0f;
		for (int i = 0; i < n; i++) {
			float abs_diff = std::abs(arr[i] - e);
			sum_x += abs_diff * abs_diff;
		}
		float a = std::sqrt(sum_x / n);

		return a;
	}


	/// <summary>
	/// Finds the ATD statistical measure and then adds the mean to it. Works over a 2d, instead of a 1d data.
	/// </summary>
	/// <param name="data"></param>
	/// <param name="threshold"></param>
	inline void threshold(std::array<std::array<float, 192>, 257>& data, float& threshold) {
		// Compute the median of the absolute values of the non-zero elements
		float median = 0.0f;
		int count = 0;
		for (int j = 0; j < NBINS_last; j++) {
			for (int i = 0; i < 192; i++) {
				float val = data[j][i];
				if (val != 0.0f) {
					median += std::abs(val);
					count++;
				}
			}
		}
		if (count == 0) {
			threshold = 0.0f;
			return;
		}
		median /= count;

		// Compute the threshold using the formula from the original implementation
		float sum = 0.0f;
		int n = 0;
		for (int j = 0; j < NBINS_last; j++) {
			for (int i = 0; i < 192; i++) {
				float val = data[j][i];
				if (!std::isnan(val) && std::isfinite(val)) {
					float diff = std::abs(val - median);
					sum += diff * diff;
					n++;
				}
			}
		}
		if (n == 0) {
			threshold = 0.0f;
			return;
		}
		threshold = std::sqrt(sum / n) + median;
	}


	/// <summary>
	/// Finds the maximum value in the array slice delineated by NBINS
	/// </summary>
	/// <param name="data"> the data to consider</param>
	/// <param name="maximum">the place to put the output</param>
	inline void find_max(const std::array<std::array<float, 192>, 257>& data, float& maximum) {
		maximum = 0;
		for (int j = 0; j < NBINS_last; j++) {
			for (int i = 0; i < 192; i++) {
				if (data[j][i] > maximum) {
					maximum = data[j][i];
				}
			}
		}
	}


	/// <summary>
	/// Takes into consideration the NBINS and provides(plausibly) the pearson coefficient for N = NBINS.
	/// </summary>
	/// <param name="X"></param>
	/// <param name="Y"></param>
	/// <returns></returns>
	inline float correlationCoefficient(const std::array<float, 257>& X, const std::array<float, 257>& Y) {
		float sum_X = std::accumulate(X.begin(), X.begin() + NBINS_last, 0.0f);
		float sum_Y = std::accumulate(Y.begin(), Y.begin() + NBINS_last, 0.0f);
		float sum_XY = std::inner_product(X.begin(), X.begin() + NBINS_last, Y.begin(), 0.0f);
		float squareSum_X = std::inner_product(X.begin(), X.begin() + NBINS_last, X.begin(), 0.0f);
		float squareSum_Y = std::inner_product(Y.begin(), Y.begin() + NBINS_last, Y.begin(), 0.0f);

		float corr = (NBINS_last * sum_XY - sum_X * sum_Y) /
			std::sqrt((NBINS_last * squareSum_X - sum_X * sum_X) *
				(NBINS_last * squareSum_Y - sum_Y * sum_Y));

		return corr;
	}



	/// <summary>
	/// Generates a logistic distribution with the endpoints substituted for the odd reflection
	/// </summary>
	inline void generate_true_logistic() {
		logit_distribution.fill({ 0.0 });
		if (NBINS_last < 4) {
			return; //note: we don't generate logistics smaller than four points

			for (int i = 0; i < NBINS_last; i++) {
				logit_distribution[i] = static_cast<float>(i) / (NBINS_last - 1);
			}
			for (int i = 1; i < NBINS_last - 1; i++) {
				logit_distribution[i] /= 1 - logit_distribution[i];
				logit_distribution[i] = std::log(logit_distribution[i]);
			}
			logit_distribution[NBINS_last - 1] = (2 * logit_distribution[NBINS_last - 2]) - logit_distribution[NBINS_last - 3];
			logit_distribution[0] = -logit_distribution[NBINS_last - 1];

			// Interpolate the elements in logit to between 0 and 1
			float min_val = logit_distribution[0];
			float max_val = logit_distribution[NBINS_last - 1];
			for (int i = 0; i < NBINS_last; i++) {
				logit_distribution[i] = (logit_distribution[i] - min_val) / (max_val - min_val);
			}
		}

	}


	/// <summary>
	/// Produces the "estimated maximum not-noise" where 1 value is 1 and all the rest are zero.
	/// Think of this in a graph space- you have your logistic which is like a flipped sigmoid,
	/// and you have your beta which is a strong indicator of data. The most unlike our wandering line is a right angle.
	/// </summary>
	inline void determine_entropy_maximum() {
		temp_257.fill({ 0 });
		temp_257[NBINS_last - 1] = 1.0f;
		MAXIMUM = 1.0f - correlationCoefficient(temp_257, logit_distribution);
	}

	/// <summary>
	/// Fast Entropy
	/// for each time-segment, sorts, interpolates to 0 and 1, and compares a subset of the 
	/// frequency bins(from 0 to NBINS_last), and derives the pearson coefficient for N = the bandpass filter size.
	/// this is then subtracted from 1.
	/// </summary>
	/// <param name="data"></param>
	inline void fast_entropy(std::array<std::array<float, 192>, 257>& data) {
		temp_257.fill({ 0 });
		for (int i = 0; i < 192; i++) {
			for (int j = 0; j < NBINS_last; j++) {
				temp_257[j] = data[j][i];
			}
			// Create a subset of the first NBINS elements of d and sort it
			std::sort(temp_257.begin(), temp_257.begin() + NBINS_last);

			float dx = temp_257[NBINS_last - 1] - temp_257[0];
			for (int j = 0; j < NBINS_last; j++) {
				temp_257[j] = (temp_257[j] - temp_257[0]) / dx;
			}

			float v = correlationCoefficient(temp_257, logit_distribution);
			if (std::isnan(v)) {
				entropy_unmasked[i] = 0.0f;
			}
			else {
				entropy_unmasked[i] = 1.0f - v;
			}
		}
	}


	/// <summary>
	/// Fast Peaks
	/// The function iteratively considers the frequency bins in each time-segment of the STFT,
	/// and for each time-segment, independently calculates some statisical measures over the set of bins within the bandpass region,
	/// and combines them with a statistic measured over the entire window, called "t".
	/// Each segment, if the corresponding entropy measurement considers it noise, is not evaluated, but is skipped over.
	/// Each "pixel" in the STFT is then either passed or not passed based on the statistical measure.
	/// It must be noted that all the prototyping was done with python, and without python, good algorithms would never even be invented.
	/// </summary>
	/// <param name="stft_">The input set of values to consider</param>
	/// <param name="mask"> the array to store the product in</param>
	inline void fast_peaks(std::array<std::array<float, 192>, 257>& stft_, std::array<std::array<float, 192>, 257>& mask) {

		//for each time bin:
		for (int each = 0; each < 192; each++) {
			if (entropy_thresholded[each] == 0) {
				continue; //skip the calculations for this row, it's masked already
			}
			//copy out the frequency components
			for (int j = 0; j < NBINS_last; j++) {
				temp_257[j] = stft_[j][each];
			}
			constant_temp = MAN(temp_257);
			test = entropy_smoothed[each] / MAXIMUM;
			test = std::abs(test - 1);
			thresh1 = (t * test);
			if (!std::isnan(thresh1)) {
				constant_temp = (thresh1 + constant_temp) / 2; //catch errors
			}
			//set the masking values
			for (int i = 0; i < NBINS_last; i++) {
				if (stft_[each][i] > constant_temp) {
					mask[each][i] = 1;
				}
			}
		}
	}



	static constexpr int NUM_STFT_FREQUENCIES = 512;
	//is this correct? should this be 256 instead?

	fftwf_plan plan_forward = fftwf_plan_dft_r2c_1d(NUM_STFT_FREQUENCIES, temp_512.data(), reinterpret_cast<fftwf_complex*>(temp_complex.data()), FFTW_ESTIMATE);

	void stft(std::array<float, 25087>& xp, const std::array<float, 512>& window, std::array<std::array<std::complex<float>, 192>, 257>& out) {

	
		int n_fft = 512;
		int hop_len = 128;

		int n_segs = 192;
		int s20 = 256;

		for (int i = 0; i < n_segs; i++) {
			int start0 = hop_len * i;
			int end0 = start0 + s20;
			int start1 = end0;
			for (int j = 0; j < s20; j++) {
				temp_512[j] = xp[start1 + j];
			}
			for (int j = s20; j < n_fft; j++) {
				temp_512[j] = xp[start0 + j];
			}
			for (int j = 0; j < 512; j++) {
				temp_512[j] = temp_512[j] * window[j];
				
			}

			fftwf_execute(plan_forward);

			for (int e = 0; e < 257; e++) {
				out[e][i] = temp_complex[e] /512.0f; //normalize to numpy, since internally we like to process using numpy normalized data.
													//this follows numpy's convention of fct = 1/512.
			}
		}
	}


	static constexpr int NUM_ISTFT_FREQUENCIES = 512;


	fftwf_plan plan_reverse = fftwf_plan_dft_c2r_1d(NUM_ISTFT_FREQUENCIES, reinterpret_cast<fftwf_complex*>(temp_complex.data()), temp_512.data(), FFTW_ESTIMATE);

	std::array<float, 8192> istft(std::array<std::array<std::complex<float>, 64>, 257>& Sx) {

		// Reuse temp, eliminate xbuf
		for (int i = 0; i < 64; i++) {
			// Perform irfft irfft
			temp_complex.fill({ 0 }); //clear  values
			for (int n = 0; n < 257; n++) {
				temp_complex[n] = Sx[n][i];
			}
			fftwf_execute(plan_reverse); 

			// Perform FFT shift for a fixed size 

			for (int j = 0;j < 256 ; j++) {
				std::swap(temp_512[j], temp_512[j + 256]);
			}

			for (int j = 0; j < 512; j++) {
				temp_512[j] = temp_512[j] * synthesis_window[j];
			}
			for (int j = 0; j < 128; j++) {
				temp_128[j] = temp_512[j];
			}
			for (int j = 0; j < 128; j++) {
				temp_128[j] += buffer[j];
			}
			// update state variables
			for (int j = 0; j < 256; j++) {//shift buffer to left 128 values
				buffer[j] = buffer[j + 128];
			}
			for (int j = 0; j < 128; j++) {
				buffer[j + 256] = 0.0; //clear the last 128 elements
			}
			for (int j = 0; j < 384; j++) {
				buffer[j] += temp_512[j + 128];
			}
			for (int j = 0; j < 128; j++) {
				output[(i * 128) + j] = temp_128[j];
			}
		}
		return output;
	}


	inline void convolve_same_entropy(std::array<float, 204>& in)
	{
		// assumes a kernel size of three and an input of INPUT_SIZE.
		//works with only odd sized kernels- but we only use odd sized kernels so its fine.
		constexpr int KERNEL_SIZE = 3;
		constexpr int INPUT_SIZE = 204;
		constexpr int N_LEFT = KERNEL_SIZE / 2;
		constexpr int N_RIGHT = KERNEL_SIZE - N_LEFT - 1;
		constexpr std::array<float, 3> kernel = { 1.0, 1.0, 1.0 };


		// Create output array
		// Perform the convolution
		std::array<float, INPUT_SIZE + (KERNEL_SIZE * 2)> ret = {};

		/*
			for i in range(n2 - n_left, n2):
			ret[idx] = innerprod(ap1[:i], ap2[-i:])
			idx += inc
		*/
		// Loop over the first portion of the output, where the filter extends beyond the left edge of the input
		for (int i = 0; i < N_LEFT; i++) {
			ret[i] = std::inner_product(in.begin(), in.begin() + KERNEL_SIZE - i, kernel.rbegin(), 0.0f);
		}
		/*
				for i in range(n1 - n2 + 1):
			ret[idx] = innerprod(ap1[i : i + n2], ap2)
			idx += inc
		*/
		// Loop over the middle portion of the output, where the filter is entirely contained within the input
		
		for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; i++) {
			ret[i + N_LEFT] = std::inner_product(in.begin() + i, in.begin() + i + KERNEL_SIZE, kernel.begin(), 0.0f);
		}


		// Loop over the last portion of the output, where the filter extends beyond the right edge of the input
		/*
				for i in range(n2 - 1, n2 - 1 - n_right, -1):
			ret[idx] = innerprod(ap1[-i:], ap2[:i])
			idx += inc

		*/
		for (int i = INPUT_SIZE - N_RIGHT; i < INPUT_SIZE; i++) {
			float sum = 0.0f;
			int kernel_idx = 0;
			for (int j = i - KERNEL_SIZE + 1; j <= i; j++) {
				sum += in[j] * kernel[kernel_idx++];
			}
			ret[i] = sum;
		}

		std::copy(ret.begin(), ret.begin() + INPUT_SIZE, in.begin());
	}


	inline void convolve_same_sawtooth(std::array<float, 222>& in)
	{
		//we only use odd sized kernels.
		constexpr int KERNEL_SIZE = 15;
		constexpr int INPUT_SIZE = 192;
		constexpr int N_LEFT = KERNEL_SIZE / 2;
		constexpr int N_RIGHT = KERNEL_SIZE - N_LEFT - 1;
		constexpr std::array<float, 15> kernel = { 0.0, 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.0, 0.85714286, 0.71428571, 0.57142857, 0.42857143, 0.28571429, 0.14285714, 0.0 };

		std::array<float, INPUT_SIZE + (KERNEL_SIZE * 2)> ret = {};

		/*
			for i in range(n2 - n_left, n2):
			ret[idx] = innerprod(ap1[:i], ap2[-i:])
			idx += inc
		*/
		// Loop over the first portion of the output, where the filter extends beyond the left edge of the input
		for (int i = 0; i < N_LEFT; i++) {
			ret[i] = std::inner_product(in.begin(), in.begin() + KERNEL_SIZE - i, kernel.rbegin(), 0.0f);
		}
		/*
				for i in range(n1 - n2 + 1):
			ret[idx] = innerprod(ap1[i : i + n2], ap2)
			idx += inc
		*/
		// Loop over the middle portion of the output, where the filter is entirely contained within the input

		for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; i++) {
			ret[i + N_LEFT] = std::inner_product(in.begin() + i, in.begin() + i + KERNEL_SIZE, kernel.begin(), 0.0f);
		}

		// Loop over the last portion of the output, where the filter extends beyond the right edge of the input
		/*
				for i in range(n2 - 1, n2 - 1 - n_right, -1):
			ret[idx] = innerprod(ap1[-i:], ap2[:i])
			idx += inc

		*/
		for (int i = INPUT_SIZE - N_RIGHT; i < INPUT_SIZE; i++) {
			float sum = 0.0f;
			int kernel_idx = 0;
			for (int j = i - KERNEL_SIZE + 1; j <= i; j++) {
				sum += in[j] * kernel[kernel_idx++];
			}
			ret[i] = sum;
		}

		std::copy(ret.begin(), ret.begin() + INPUT_SIZE, in.begin());
	}
	
	inline void numpy_entropy_smooth(std::array<float, 192>& in, std::array<float, 192>& out) {
		constexpr int PADDING_SIZE = 6;
		constexpr int INPUT_SIZE = 192;

		std::array<float, 192 + PADDING_SIZE * 2> padded = {};
		for (int i = PADDING_SIZE; i < INPUT_SIZE + PADDING_SIZE; i++) {
			padded[i] = in[i - PADDING_SIZE];
		}
		convolve_same_entropy(padded);

		for (int i = PADDING_SIZE; i < INPUT_SIZE + PADDING_SIZE; i++) {
			out[i - PADDING_SIZE] = padded[i]/3.0f;
		}
	}
	
	inline void numpy_sawtooth_smooth(std::array<float, 192>& in) {
		constexpr int PADDING_SIZE = 15;
		constexpr int INPUT_SIZE = 192;

		std::array<float, 192 + PADDING_SIZE * 2> padded = {};
		for (int i = PADDING_SIZE; i < INPUT_SIZE + PADDING_SIZE; i++) {
			padded[i] = in[i - PADDING_SIZE];
		}
		convolve_same_sawtooth(padded);

		for (int i = PADDING_SIZE; i < INPUT_SIZE + PADDING_SIZE; i++) {
			in[i - PADDING_SIZE] = padded[i];
		}
	}



	inline void convolve_same_frequency(std::array<float, 257 + 6>& in)
	{
		// remember, for frequency, we do 3,3 padding of 257 but filter of 13.
		constexpr int KERNEL_SIZE = 13;
		constexpr int INPUT_SIZE = 257 + 6;

		constexpr std::array<float, 13> kernel = { 1.0,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 ,1.0 };

		// Compute the number of left and right elements
		const int N_LEFT = KERNEL_SIZE / 2;
		const int N_RIGHT = KERNEL_SIZE - N_LEFT - 1;

		// Create output array
		// Perform the convolution
		std::array<float, INPUT_SIZE + (KERNEL_SIZE * 2)> ret = {};

		/*
			for i in range(n2 - n_left, n2):
			ret[idx] = innerprod(ap1[:i], ap2[-i:])
			idx += inc
		*/
		// Loop over the first portion of the output, where the filter extends beyond the left edge of the input
		for (int i = 0; i < N_LEFT; i++) {
			ret[i] = std::inner_product(in.begin(), in.begin() + KERNEL_SIZE - i, kernel.rbegin(), 0.0f);
		}
		/*
				for i in range(n1 - n2 + 1):
			ret[idx] = innerprod(ap1[i : i + n2], ap2)
			idx += inc
		*/
		// Loop over the middle portion of the output, where the filter is entirely contained within the input

		for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; i++) {
			ret[i + N_LEFT] = std::inner_product(in.begin() + i, in.begin() + i + KERNEL_SIZE, kernel.begin(), 0.0f);
		}

		// Loop over the last portion of the output, where the filter extends beyond the right edge of the input
		/*
				for i in range(n2 - 1, n2 - 1 - n_right, -1):
			ret[idx] = innerprod(ap1[-i:], ap2[:i])
			idx += inc

		*/
		for (int i = INPUT_SIZE - N_RIGHT; i < INPUT_SIZE; i++) {
			float sum = 0.0f;
			int kernel_idx = 0;
			for (int j = i - KERNEL_SIZE + 1; j <= i; j++) {
				sum += in[j] * kernel[kernel_idx++];
			}
			ret[i] = sum;
		}

		std::copy(ret.begin(), ret.begin() + INPUT_SIZE, in.begin());
	}

	inline void convolve_same_time(std::array<float, 192 + 26>& in)
	{
		// remember, for frequency, we do 13,13 padding of 192 but filter of 3.
		constexpr int KERNEL_SIZE = 3;
		constexpr int INPUT_SIZE = 192 + 26;

		constexpr std::array<float, 3> kernel = { 1.0,1.0 ,1.0 };

		// Compute the number of left and right elements
		const int N_LEFT = KERNEL_SIZE / 2;
		const int N_RIGHT = KERNEL_SIZE - N_LEFT - 1;

		// Create output array
		// Perform the convolution
		std::array<float, INPUT_SIZE + (KERNEL_SIZE * 2)> ret = {};

		/*
			for i in range(n2 - n_left, n2):
			ret[idx] = innerprod(ap1[:i], ap2[-i:])
			idx += inc
		*/
		// Loop over the first portion of the output, where the filter extends beyond the left edge of the input
		for (int i = 0; i < N_LEFT; i++) {
			ret[i] = std::inner_product(in.begin(), in.begin() + KERNEL_SIZE - i, kernel.rbegin(), 0.0f);
		}
		/*
				for i in range(n1 - n2 + 1):
			ret[idx] = innerprod(ap1[i : i + n2], ap2)
			idx += inc
		*/
		// Loop over the middle portion of the output, where the filter is entirely contained within the input

		/*for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; i++) {
			ret[i + N_LEFT] = std::inner_product(in.begin() + i, in.begin() + i + KERNEL_SIZE, kernel.begin(), 0.0f);
		}*/

		for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; i++) {
			ret[i + N_LEFT] = std::inner_product(in.begin() + i, in.begin() + i + KERNEL_SIZE, kernel.begin(), 0.0f);
		}

		// Loop over the last portion of the output, where the filter extends beyond the right edge of the input
		/*
				for i in range(n2 - 1, n2 - 1 - n_right, -1):
			ret[idx] = innerprod(ap1[-i:], ap2[:i])
			idx += inc

		*/
		for (int i = INPUT_SIZE - N_RIGHT; i < INPUT_SIZE; i++) {
			float sum = 0.0f;
			int kernel_idx = 0;
			for (int j = i - KERNEL_SIZE + 1; j <= i; j++) {
				sum += in[j] * kernel[kernel_idx++];
			}
			ret[i] = sum;
		}


		std::copy(ret.begin(), ret.begin() + INPUT_SIZE, in.begin());
	}

	/// <summary>
	/// Performs same mode convolution in the time domain using a triangular, symmetric filter of 15 elements,
	/// which summarize back to the input *7. The product is then divided by 7 and stored in the output array.
	/// we leave the exercise to the reader of improving the efficiency of the behavior here by in-place copying
	/// the contents of a 2d array into a 1d array(like a ravel or flatten), and then appropriately
	/// convolving the 1d, and then copying the contents back out. this will require inserting at least
	/// filter size *2 elements between each row in the 1d to allow the tail end of one row not to alter the next.
	/// </summary>
	/// <param name="stft_real">The input</param>
	/// <param name="smoothed">The output</param>
	inline void sawtooth_convolve(std::array<std::array<float, 192>, 257>& stft_real, std::array<std::array<float, 192>, 257>& smoothed) {

		std::array<float, 192> temp = {};

		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				temp[j] = stft_real[i][j];
			}
			numpy_sawtooth_smooth(temp);
			for (int j = 0; j < 192; j++) {
				smoothed[i][j] = temp[j] / 7.0f;
			}
		}
	}






	/// <summary>
	/// Finds the longest streak of ones in an array of 192 integers and returns it
	/// </summary>
	/// <param name="nums"></param>
	/// <returns></returns>
	inline int longestConsecutive(std::array<int, 192>& nums) {
		int curr_streak = 0;
		int prevstreak = 0;
		for (int i = 0; i < 192; i++) {
			if (nums[i] == 1) {
				curr_streak++;
			}
			else if (nums[i] == 0) {
				if (curr_streak > prevstreak) {
					prevstreak = curr_streak;
				}
				curr_streak = 0;
			}
		}
		return (curr_streak > prevstreak) ? curr_streak : prevstreak;
	}


	/// <summary>
	/// Given the input, finds islands under a certain size and removes spurious discongruities
	/// </summary>
	/// <param name="a"> array to process</param>
	/// <param name="value"> value to replace</param>
	/// <param name="threshold">size of island</param>
	/// <param name="replace"> replacement values</param>
	inline void remove_outliers(std::array<int, 192>& a, int value, int threshold, int replace) {
		int first = 0;
		int end = 0;
		int index = 0;
		while (first < 192) {
			if (a[first] == value) {
				index = first;
				while (index < 192 && a[index] == value) {
					index += 1;
				}
				end = index;
				if (end - first + 1 < threshold) {
					for (int i = first; i < end; i++) {
						a[i] = replace;
					}
				}
				first = end;
			}
			else {
				index = first;
				while (index < 192 && a[index] != value) {
					index += 1;
				}
				first = index;
			}
		}
	}

	inline void process_entropy() {

		fast_entropy(stft_real);
		numpy_entropy_smooth(entropy_unmasked, entropy_smoothed);

		entropy_thresholded.fill({ 0 });


		float ent_max = 0;


		for (int i = 0; i < 192; i++) {
			if (entropy_smoothed[i] > ent_max) {
				ent_max = entropy_smoothed[i];
			}
		}



		for (int i = 0; i < 192; i++) {
			if (entropy_smoothed[i] > CONST_last) {
				entropy_thresholded[i] = 1;
				if ((i > 30) && (i < 160)) { //same thing as numpy slice of [32:128+32].. right?
					count++;
				}
			}
		}

		if ((count > 22 || longestConsecutive(entropy_thresholded) > 16)) {
			flag = 2;
			remove_outliers(entropy_thresholded, 0, 6, 1);
			remove_outliers(entropy_thresholded, 1, 2, 0);
		}
		count = 0;
	}

	inline void smooth_and_mask() {



		sawtooth_convolve(stft_real, smoothed);


		// Transpose data into scratch


		fast_peaks(smoothed, previous);

		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				if (previous[i][j] == 0) {
					stft_real[i][j] = 0.0;
				}
			}
		}

		find_max(stft_real, multiplier);
		multiplier = multiplier / initial;
		if (multiplier > 1) { multiplier = 1; }


		sawtooth_convolve(stft_real, smoothed);


		fast_peaks(smoothed, smoothed);

		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				initial = smoothed[i][j] * multiplier;
				if (previous[i][j] > initial) {
					previous[i][j] = initial;
				}
			}
		}

		sawtooth_convolve(previous, previous);



		vertical.fill({ 0 }); //clear corner padding
		horizontal.fill({ 0 });

		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				//apply the mask
				//remember that we include padding at both ends of filter size for intermediate products.
				vertical[i + FREQ_PAD][j + TIME_PAD] = previous[i][j];
				horizontal[i + FREQ_PAD][j + TIME_PAD] = previous[i][j];

			}
		}


		//perform iterative 2d smoothing.



		//now, we leave as an exercise to the reader an optimization to the below operations which is easily written in python but not here.
		//Taking the below behavior, copy each of the padded rows into an offset for a 1d array.
		//likewise, do the same with the transpose of the 2d.
		//perform the convolution on the 1d for each, appropriately switching the filter size and the division.
		//reverse the reshape back to the 2d, and for the transpose, re-transpose the product back.
		//add the two together, dividing by two, then duplicate the first into the second.
		//you now have the same behavior as below, but using a two 1d convolutional steps for each iteration,
		//working over a much larger array, but with linear behavior that can automatically vectorize.
		//copying is cheap on modern processors- convolutional optimization is more expensive.
		//in python we already do this for the python version of the loop.


		for (int e = 0; e < 2; e++) {

			//do first dimension first

			for (int i = 0; i < 192 + TIME_PAD * 2; i++) {//iterating over time
				//each iteration, iterate through the working area + the padding area.
				//start at zero, because our infill starts at 3, and therefore the padding area includes the area before.
				// restrict by NBINS_last because too much would be overkill.
				//at most this is the entire array, at the least, this is the padding and the filling.

				frequencywise_storage.fill({ 0 }); //clear the working memory

				for (int j = 0; j < NBINS_last + FREQ_PAD * 2; j++) {
					//infill our temporary memory with the persistent padding and the data
					frequencywise_storage[j] = vertical[j][i];
				}
				convolve_same_frequency(frequencywise_storage);
				for (int j = 0; j < NBINS_last + FREQ_PAD * 2; j++) {
					//infill our temporary memory with the persistent padding and the data
					vertical[j][i] = frequencywise_storage[j] / 13.0f;
				}
			}

			for (int i = 0; i < NBINS_last + FREQ_PAD * 2; ++i) {
				auto& row = horizontal[i];
				convolve_same_time(row);
			}




			for (int i = 0; i < NBINS_last + FREQ_PAD * 2; i++) {
				for (int j = 0; j < 192 + TIME_PAD * 2; j++) {
					//apply the mask, conserving the padding
					vertical[i][j] = (vertical[i][j] + horizontal[i][j]) / 2.0f;
					horizontal[i][j] = vertical[i][j];

				}
			}

		}

		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				//apply the smoothing, slicing out of our array
				previous[i][j] = vertical[i + TIME_PAD][j + FREQ_PAD];

			}
		}
	}


public:

	~Filter()
	{
		fftwf_destroy_plan(plan_forward); //must be manually cleared
		fftwf_destroy_plan(plan_reverse); //must be manually cleared

	}

	/// <summary>
	/// Sets the value of the temporary placeholder for the Constant
	/// </summary>
	/// <param name="val"></param>
	void setConstant(float val) {
		CONST_1 = val;
	}

	/// <summary>
	/// sets the value of the temporary placeholder for the number of bins to consider
	/// </summary>
	/// <param name="val"></param>
	void set_NBINS(int val) {
		if (val > 0) {
			if (val < (hop_size * 2) + 2) {
				NBINS_1 = val;
			}
			else {
				std::cout << "bandpass limit  was corrected to " << ((hop_size * 2) + 1) << " bandpass greater than hop size*2 + 1 not allowed" << std::endl;
				NBINS_1 = ((hop_size * 2) + 1);
			}
		}
	}

	void setSampleRate(int val) {
		int flag = 1;
		// Check if the given value is one of the supported sample rates.
		if (val == 6000 || val == 12000 || val == 24000 || val == 48000) {
			flag = 0;
		}
		
			// Round up to the nearest supported sample rate, if necessary, and set the settings.
		if (val < 6001) {
				sample_rate = 6000;
				N_FFT = 64;
				hop_size = 32;
			}
		else if (val < 12001) {
				sample_rate = 12000;
				N_FFT = 128;
				hop_size = 64;
			}
		else if (val < 24001) {
				sample_rate = 24000;
				N_FFT = 256;
				hop_size = 128;
			}
		else {
				sample_rate = 48000;
				N_FFT = 512;
				hop_size = 128;
			}
		if (flag == 1) {
			std::cout << "Sample rate was corrected to " << sample_rate << ", sample rates not a divisor of 48k greater than 3k not supported" << std::endl;

		}
	}


	//TODO: find ways to merge the uses of the above so that a mimimum in working memory can be utilized

	std::array<float, 8192> process(std::array<float, 8192> input) {


		if (CONST_last != CONST_1) {
			if (CONST_1 > 1) { CONST_1 = 1.0; }//make sure sane inputs
			if (CONST_1 < 0) { CONST_1 = 0.001; }
			CONST_last = CONST_1; //if const-1 is changed this means it will only update once per cycle

		}

		if (NBINS_1 != NBINS_last) {//same thing for nbins- only considered once per cycle
			if (NBINS_1 > 257) { NBINS_1 = 257; }//make sure sane inputs
			if (NBINS_1 < 5) {
				std::cout << "bandpass limit  was corrected to 5 bands , bandpass less  than 5 not allowed" << std::endl;
				NBINS_1 = 5; }//you cant use less than 5 bins!
			if (NBINS_1 == 37) {//let's fill the std::array
				NBINS_last = NBINS_1;
				std::copy(std::begin(logit_37), std::end(logit_37), std::begin(logit_distribution));
				MAXIMUM = 0.6122167f;
			}
			else {
				//generate and set logit, entropy maximum
				NBINS_last = NBINS_1;
				generate_true_logistic();
				determine_entropy_maximum();

			}
		}


		//in this function we are only to allocate and do reference based manipulation of data- never copy or move. 
		for (int i = 0; i < 8192 * 2; i++) {
			audio[i] = audio[i + 8192];
		}
		//rapidly copy in the audio
		std::copy(std::begin(input), std::end(input), std::end(audio) - 8192);

		for (int i = 256; i < 25087 - 255; i++) {
			audio_padded[i] = audio[i - 256];
		}
		for (int i = 0; i < 256; i++) {
			audio_padded[i] = audio[257 + 2 * 256 - i - 1];
		}
		for (int i = 0; i < 255; i++) {
			audio_padded[25087 - 255 + i] = audio[audio.size() - 256 - i - 1];
		}
		stft(audio_padded, shifted_logistic_window, stft_complex);
		// Copy the first 37 rows of stft_complex to stft_real
		for (int i = 0; i < NBINS_last; i++) {
			for (int j = 0; j < 192; j++) {
				stft_real[i][j] = std::abs(stft_complex[i][j]);
			}
		}

		process_entropy();
		if (flag == 2) {

			threshold(stft_real, t);
			find_max(stft_real, initial);

			smooth_and_mask();


			stft(audio_padded, shifted_hann_window, stft_complex);

			for (int i = 0; i < NBINS_last; i++) {
				for (int j = 0; j < 64; j++) {
					//apply the mask

					stft_output[i][j] = stft_complex[i][j + 63] * 512.0f;//normalize to what fftw expects - needs to match the input normalization!
					stft_output[i][j] = stft_output[i][j] * previous[i][j + 63];
				}
			}

			flag = 0; //update the flag because we are good to go
			return  istft(stft_output);
		}

		if (flag == 0) {
			// Compute product using the residual buffer since on the last round it contained data

			flag = 1; //update the flag since we processed zeros
			return  istft(stft_output);
		}

		//in the final case, we no longer need to process a residual buffer,
		//since the buffer was zeros on the last run, so no istft call is needed here.
		// Return zero std::array
		return empty;

	};
};





////////////code below here is just for testing purposes
void generate_sine_wave(std::array<float, 8192>& data, float freq, float phase, float amplitude) {
	const float sample_rate = 48000;
	const float delta_phase = 2.0f * M_PI * freq / sample_rate;

	for (int i = 0; i < 8192; i++) {
		data[i] += amplitude * sin(delta_phase * i + phase);
	}
}
#ifndef M_PI_2
#define M_PI_2 (M_PI / 2.0)
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4.0)
#endif




#include <ctime>

int main() {
	Filter my_filter;
	std::array<float, 8192> demo = { 0 };
	generate_sine_wave(demo, 440.0f, 0.0f, 1.0f);
	std::array<float, 8192> output = { 0 };

	clock_t start_time, end_time;
	start_time = clock(); // get start time
	
	for (int i = 0; i < 20; i++) {
		output = my_filter.process(demo); // execute the function
	}

	end_time = clock(); // get end time
	float duration = (float)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0; // calculate duration in milliseconds
	std::cout << "Total execution time: " << duration << " milliseconds" << std::endl;
	system("pause");


	return 0;
}

