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

	size_t nob = 24;
	std::cout << "-------------------------------------------- Loading Data ----------------------------" << std::endl;
	auto C = xerus::Tensor({nob,nob});
	auto V_AO = xerus::Tensor({nob,nob,nob,nob});
	auto H_AO = xerus::Tensor({nob,nob});


	std::string name = "Data/cc-pVDZ"+std::to_string(nob)+".tensor";
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

  value_t v = 0.0;
	for(size_t i = 0; i < 5; ++i)
	    for(size_t j = 0; j < 5; ++j)
	    	if (i == j){
	    		std::cout << H_MO[{i,j}] << " " ;
	    		v += H_MO[{i,j}];
	    	}
  std::cout << std::endl;


	std::cout << "-------------------------------------------- Building TT Operator (Hamiltonian) ------" << std::endl;
	auto eps = 10e-8;

	size_t dim = 48; // 16 electron, 8 electron pairs

	auto hamiltonian = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
	// 1 e operator
	for (size_t i = 0; i < dim; i++){
		std::cout << i << std::endl;
		for (size_t j = 0; j < dim; j++){
			if (i % 2 == j % 2){
				value_t val = H_MO[{i / 2, j / 2}];
				if (std::abs(val) < 10e-10)
					continue;
				hamiltonian += val * return_one_e_ac(i,j,dim);
			}
		}
	}
	for (size_t a = 0; a < dim; a = a + 2){
		std::cout << "a     = " << a << std::endl;
		for (size_t b = 0; b < dim; b = b + 2){
			std::cout << "a,b   = " << a  << "," << b<< std::endl;
			clock_t start2 = std::clock();

			for (size_t c = 0; c < dim; c = c + 2){
				clock_t start = std::clock();
				std::cout << "a,b,c = " << a  << "," << b<< "," << c << std::endl;
				hamiltonian.canonicalized = false;
				for (size_t d = 0; d <= c ; d = d + 2){
					value_t val = V_MO[{a / 2, b / 2, c / 2, d / 2}];
					if (std::abs(val) < eps )//|| a == b || c: == d) // TODO check i == j || k == l
						continue;
					hamiltonian += val * return_two_e_ac_full(a,b,c,d,dim);
				}
				hamiltonian.move_core(0);


				clock_t end = std::clock();
				auto elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
				std::cout << "Time elapsed for iteration: " << elapsed_secs << " sec" << std::endl;
			}

			clock_t end2 = std::clock();
			auto elapsed_secs2 = double(end2 - start2) / CLOCKS_PER_SEC;
			std::cout << "Time elapsed for outer iteration: " << elapsed_secs2 << " sec" << std::endl;
		}
	}
  std::cout << "Hamiltonian " << hamiltonian.frob_norm() << std::endl;


  std::string name2 = "Data/hamiltonian" + std::to_string(dim) + ".ttoperator";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,hamiltonian,xerus::misc::FileFormat::BINARY);
	write.close();

	std::cout << "-------------------------------------------- Building TT Operator (particle number) --" << std::endl;

	auto particle_number = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
	for (size_t i = 0; i < dim; i++){
			value_t val = 1.0;
			particle_number += val * return_one_e_ac(i,i,dim);
	}

	std::string name3 = "Data/particle_number" + std::to_string(dim) + ".ttoperator";
	std::ofstream write2(name3.c_str() );
	xerus::misc::stream_writer(write2,particle_number,xerus::misc::FileFormat::BINARY);
	write2.close();
	std::cout << "particle_number " << particle_number.frob_norm() << std::endl;


	std::cout << "-------------------------------------------- Finished Building  TT operators ---------" << std::endl;
	std::cout << "-------------------------------------------- Some checks -----------------------------" << std::endl;
	auto phi = xerus::TTTensor(std::vector<size_t>(dim,2));
	for(size_t i = 0; i < 2; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[1] = 1.0;
		phi.set_component(i,tmp);
	}
	for(size_t i = 2; i < dim; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[0] = 1.0;
		phi.set_component(i,tmp);
	}
	xerus::Tensor exp_pn, exp_hamilton;
	exp_hamilton() = hamiltonian(ii/2,jj/2) * phi(ii&0) * phi(jj&0);
	exp_pn() = particle_number(ii/2,jj/2) * phi(ii&0) * phi(jj&0);

	std::cout << "(P*phi,phi) =  " << exp_pn[0] << std::endl;
	std::cout << "(H*phi,phi) =  " << exp_hamilton[0] << std::endl;

//	auto test = return_two_e_ac2(0,1,2,3,5);
//	for ( size_t i = 0; i < 5;++i){
//		std::cout <<"Component "<< i << " = \n" << test.get_component(i) << std::endl;
//	}

	auto  Ehf = 2 * H_MO[{0, 0}] +  V_MO[{0,0,0,0}];
	std::cout << std::setprecision(16)<< "E_HF = " << Ehf + 0.71510433908 << std::endl;
	std::cout << std::setprecision(16)<< "E_HF = " << exp_hamilton[0] +  2*V_MO[{0,0,0,0}] + 0.71510433908 << std::endl;
	std::cout << "H_MO[{0, 0}] = " << H_MO[{0, 0}] << std::endl;
	std::cout << "V_MO[{0,0,0,0}] = " << V_MO[{0,0,0,0}] << std::endl;

	auto cr1 = return_create(1,dim);
	auto cr2 = return_create(0,dim);
	auto an1 = return_annil(1,dim);
	auto an2 = return_annil(0,dim);
	xerus::Tensor res;
	res() =  cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2) * phi(ii&0) * phi(mm&0);

	std::cout << "(a*phi,phi) =  " << res[0] << std::endl;
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
