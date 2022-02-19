#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
using namespace xerus;
using namespace Eigen;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);

int main() {

	size_t nob = 18;
	std::cout << "-------------------------------------------- Loading Data ----------------------------" << std::endl;
	auto C = xerus::Tensor({nob,nob});
	auto V_AO = xerus::Tensor({nob,nob,nob,nob});
	auto H_AO = xerus::Tensor({nob,nob});


	std::string name = "Data/aug-cc-pVDZ"+std::to_string(nob)+".tensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,V_AO,xerus::misc::FileFormat::BINARY);
	read.close();

	name = "Data/oneParticleOperator"+std::to_string(nob)+".csv";
	Mat H_Mat = load_csv<Mat>(name);
	name = "Data/hartreeFockEigenvectors"+std::to_string(nob)+".csv";
	Mat C_Mat = load_csv<Mat>(name);
	name = "Data/oneParticleOperator_overlap"+std::to_string(nob)+".csv";
	Mat S_Mat = load_csv<Mat>(name);

	std::cout << " C size " << C_Mat.rows() << "x" << C_Mat.cols() << std::endl;
	std::cout << " H size " << H_Mat.rows() << "x" << H_Mat.cols() << std::endl;
	std::cout << " V size " << V_AO.dimensions[0] << "x" << V_AO.dimensions[1] << "x"<< V_AO.dimensions[2] << "x"<< V_AO.dimensions[3] << std::endl;
  for(size_t i = 0; i < nob; ++i){
    for(size_t j = 0; j < nob; ++j){
    	C[{i,j}] = C_Mat(i,j);
    	H_AO[{i,j}] = H_Mat(i,j);
    }
  }

	std::cout << "-------------------------------------------- Basis Transformation --------------------" << std::endl;

  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll,mm;
	auto V_MO = xerus::Tensor({nob,nob,nob,nob});
	auto H_MO = xerus::Tensor({nob,nob});

  V_MO(i1,i2,i3,i4) =  V_AO(ii,jj,kk,ll) * C(ii,i1)*C(jj,i2)*C(kk,i3)*C(ll,i4);
  H_MO(i1,i2) = C(ii,i1)*C(jj,i2)*H_AO(ii,jj);

	std::cout << "V_AO norm " << V_AO.frob_norm() << std::endl;
	std::cout << "V_MO norm " << V_MO.frob_norm() << std::endl;
	std::cout << "H_AO norm " << H_AO.frob_norm() << std::endl;
	std::cout << "H_MO norm " << H_MO.frob_norm() << std::endl;

	std::cout << "-------------------------------------------- Building TT Operator (Hamiltonian Orig) --" << std::endl;

	size_t dim = 10; // 16 electron, 8 electron pairs
	clock_t begin1 = std::clock();
	auto hamiltonian_orig = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
	// 1 e operator
	for (size_t i = 0; i < dim; i++){
		for (size_t j = 0; j < dim; j++){
			if (i % 2 == j % 2){
				value_t val = H_MO[{i / 2, j / 2}];
				if (std::abs(val) < 10e-10)
					continue;
				hamiltonian_orig += val * return_one_e_ac(i,j,dim);
			}
		}
	}
	// 2 e operator
	size_t count1 = 0;
		for (size_t a = 0; a < dim; a = a + 2){
			std::cout << "a = " << a  << std::endl;
			for (size_t b = 0; b < dim; b = b + 2){
				for (size_t c = 0; c < dim; c = c + 2){
					for (size_t d = 0; d <= c ; d = d + 2){
						count1++;
						value_t val = V_MO[{a / 2, b / 2, c / 2, d / 2}]; // TODO Check Factor 0.5 and use symmetries from libint addition
						if (std::abs(val) < 10e-10 )//|| a == b || c: == d) // TODO check i == j || k == l
							continue;
						hamiltonian_orig += val * return_two_e_ac_full(a,b,c,d,dim); //TODO check minus sign!!

					}
				}
			}
			//hamiltonian_new.round(eps);
		}
	clock_t end1 = std::clock();
  std::cout << "Hamiltonian " << hamiltonian_orig.frob_norm() << std::endl;


	std::cout << "-------------------------------------------- Building TT Operator (Hamiltonian New) --" << std::endl;
	auto eps = 10e-8;
	clock_t begin2 = std::clock();
	auto hamiltonian_new = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
	// 1 e operator
	for (size_t i = 0; i < dim; i++){
		for (size_t j = 0; j < dim; j++){
			if (i % 2 == j % 2){
				value_t val = H_MO[{i / 2, j / 2}];
				if (std::abs(val) < eps)
					continue;
				hamiltonian_new += val * return_one_e_ac(i,j,dim);
			}
		}
	}
	size_t count = 0;
	for (size_t a = 0; a < dim; a = a + 2){
		hamiltonian_new.canonicalized = false;
		std::cout << "a = " << a  << std::endl;
		for (size_t b = 0; b < dim; b = b + 2){
			for (size_t c = 0; c < dim; c = c + 2){
				for (size_t d = 0; d <= c ; d = d + 2){
					count++;
					value_t val = V_MO[{a / 2, b / 2, c / 2, d / 2}]; // TODO Check Factor 0.5 and use symmetries from libint addition
					if (std::abs(val) < eps )//|| a == b || c: == d) // TODO check i == j || k == l
						continue;
					hamiltonian_new += val * return_two_e_ac_full(a,b,c,d,dim); //TODO check minus sign!!
				}
			}
		}
		hamiltonian_new.move_core(0);

		//hamiltonian_new.round(eps);
	}
	clock_t end2 = std::clock();

  std::cout << "Hamiltonian " << hamiltonian_new.frob_norm() << std::endl;
	std::cout << "-------------------------------------------- Finished Building  TT operators ---------" << std::endl;
	std::cout << "-------------------------------------------- Some checks -----------------------------" << std::endl;
  std::cout << "Norm orig - new =  " << (hamiltonian_new - hamiltonian_orig).frob_norm() << std::endl;
  auto elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
  auto elapsed_secs2 = double(end2 - begin2) / CLOCKS_PER_SEC;
  std::cout << "Time elapsed for Original Hamiltonian: " << elapsed_secs1 << " sec" << std::endl;
  std::cout << "Time elapsed for New Hamiltonian     : " << elapsed_secs2 << " sec" << std::endl;
  std::cout << "Speed up                             : " << elapsed_secs2 / elapsed_secs1  << std::endl;
  std::cout << "Count                                : " << count <<  std::endl;
  std::cout << "Count                                : " << count1 <<  std::endl;


	return 0;
}


//Creation of Operators
xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this


	xerus::Index i1,i2,jj, kk, ll;
	auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto annhil = xerus::Tensor({2,2});
	annhil[{0,1}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? annhil : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		a_op.set_component(m, res);
	}
	return a_op;
}

xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
	xerus::Index i1,i2,jj, kk, ll;
	auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto create = xerus::Tensor({2,2});
	create[{1,0}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? create : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		c_op.set_component(m, res);
	}
	return c_op;
}

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d){ // TODO write tests for this
	auto cr = return_create(i,d);
	auto an = return_annil(j,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk;
	res(ii/2,jj/2) = cr(ii/2,kk/2) * an(kk/2, jj/2);
	return res;
}

xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test
	auto cr1 = return_create(i,d);
	auto cr2 = return_create(j,d);
	auto an1 = return_annil(k,d);
	auto an2 = return_annil(l,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll,mm;
	res(ii/2,mm/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2);
	return res;
}

xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim){
	auto res = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	if (a!=b && c!=d)
		res += return_two_e_ac(a,b,d,c,dim);
		res += return_two_e_ac(a+1,b+1,d+1,c+1,dim);
	if (c != d)
		res += return_two_e_ac(a+1,b,d,c+1,dim);
	res += return_two_e_ac(a,b+1,d+1,c,dim);


	return res;
}



template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}
