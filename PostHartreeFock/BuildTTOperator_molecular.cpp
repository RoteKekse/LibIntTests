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

xerus::TTOperator return_spin_op(size_t i, size_t d, bool transpose, bool up);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, bool s, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, bool s1, bool s2, size_t d);
//xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);

int main() {
	/*
	 * !!!!! Change Here !!!!
	 */
	size_t dim = 10; // 16 electron, 8 electron pairs !!!!! Change Here !!!!
	size_t nob = 25;
	auto eps = 10e-12;

	std::cout << "-------------------------------------------- Loading Data ----------------------------" << std::endl;
	auto C = xerus::Tensor({nob,nob});
	auto V_AO = xerus::Tensor({nob,nob,nob,nob});
	auto H_AO = xerus::Tensor({nob,nob});


	std::string name = "Data/cc-pvdz_bauschlicher"+std::to_string(nob)+".tensor";
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

	std::cout << "-------------------------------------------- Building TT Operator (Hamiltonian) ------" << std::endl;

	auto hamiltonian = xerus::TTOperator(std::vector<size_t>(2*dim,4)); // TT operator initialized with 0
	// 1 e operator
	for (size_t i = 0; i < dim; i++){
		std::cout << i << std::endl;
		for (size_t j = 0; j < dim; j++){
			value_t val = H_MO[{i , j}];
			if (std::abs(val) < eps)
				continue;
			for(bool s : {true, false}){
				hamiltonian += val * return_one_e_ac(i,j,s,dim);

			}
		}
	}

	clock_t start_full = std::clock();
	for (size_t i = 0; i < dim; ++i){
		std::cout << "i     = " << i << std::endl;
		for (size_t j = 0; j < dim; ++j){
			std::cout << "i,j   = " << i  << "," << j<< std::endl;
			clock_t start2 = std::clock();
			for (size_t k = 0; k < dim; ++k){
				clock_t start = std::clock();
				std::cout << "i,j,k = " << i  << "," << j<< "," << k << std::endl;
				hamiltonian.canonicalized = false;
				auto tmp = xerus::TTOperator(std::vector<size_t>(2*dim,4)); // TT operator initialized with 0
				for (size_t l = 0; l <= k ; ++l){ //use symmetry!!
					value_t val = V_MO[{i,j,k,l}];
					if (l == k) val = val * 0.5;
					if (std::abs(val) < eps )
						continue;
					for(bool s1 : {true, false}){
						for(bool s2 : {true, false}){
							tmp +=  val * return_two_e_ac(i,j,k,l,s1,s2,dim); //TODO check order!!
						}
					}
				}
				hamiltonian += tmp;
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
	clock_t end_full = std::clock();
	auto elapsed_secs_full = double(end_full - start_full) / CLOCKS_PER_SEC;
	std::cout << "Time elapsed for full iteration: " << elapsed_secs_full << " sec" << std::endl;

  std::cout << "Hamiltonian " << hamiltonian.frob_norm() << std::endl;


  std::string name2 = "Data/hamiltonian" + std::to_string(dim) + ".ttoperator";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,hamiltonian,xerus::misc::FileFormat::BINARY);
	write.close();

	std::cout << "-------------------------------------------- Finished Building  TT operators ---------" << std::endl;


	std::cout << "-------------------------------------------- Tests  TT operators ---------" << std::endl;
	size_t dd = 5;
	auto t1 = return_two_e_ac(0,2,3,4,true,false,dd);

	for (size_t ind = 0; ind < dd; ++ind){
		std::cout << t1.get_component(ind) << std::endl;
	}
	xerus::Index iii,jjj;
	auto phi = xerus::TTTensor(std::vector<size_t>(dim,4));
		for(size_t i = 0; i < 5; ++i){
			auto tmp = xerus::Tensor({1,4,1});
			tmp[{0,3,0}] = 1.0;
			phi.set_component(i,tmp);
		}
		for(size_t i = 5; i < dim; ++i){
			auto tmp = xerus::Tensor({1,4,1});
			tmp[{0,0,0}] = 1.0;
			phi.set_component(i,tmp);
		}
		xerus::Tensor exp_pn, exp_hamilton;
		exp_hamilton() = hamiltonian(iii/2,jjj/2) * phi(iii&0) * phi(jjj&0);
		std::cout << "(H*phi,phi) =  " << exp_hamilton[0] << std::endl;
		std::cout << "V_MO =  " << V_MO[{0,0,0,0}] << std::endl;


	return 0;
}

//Spin Up Operator
xerus::TTOperator return_spin_op(size_t i, size_t d, bool transpose, bool up){ // TODO write tests for this
	xerus::Index i1,i2,jj, kk, ll;
	auto s_op = xerus::TTOperator(std::vector<size_t>(2*d,4));
	auto id = xerus::Tensor::identity({4,4});
	auto z = xerus::Tensor::identity({4,4});
	z[{1,1}] = -1.0;
	z[{2,2}] = -1.0;
	auto spin = xerus::Tensor({4,4});
	if (up){
		spin[{0,2}] = 1.0;
		spin[{1,3}] = 1.0;
	} else {
		spin[{0,1}] = 1.0;
		spin[{2,3}] = -1.0;
	}
	if (transpose) spin(i1,i2) = spin(i2,i1);
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? z : (m == i ? spin : id );
		tmp.reinterpret_dimensions({1,4,4,1});
		tmp.use_sparse_representation(10e-15);
		s_op.set_component(m, tmp);
	}
	return s_op;
}


xerus::TTOperator return_one_e_ac(size_t i, size_t j, bool s, size_t d){ // TODO write tests for this
	auto ci = return_spin_op(i,d,true,s);
	auto cj = return_spin_op(j,d,false,s);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk;
	res(ii/2,jj/2) = ci(ii/2,kk/2) * cj(kk/2, jj/2);
	return res;
}

xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, bool s1, bool s2, size_t d){ //todo test
	auto ci = return_spin_op(i,d,true,s1);
	auto cj = return_spin_op(j,d,true,s2);
	auto ck = return_spin_op(k,d,false,s1);;
	auto cl = return_spin_op(l,d,false,s2);;
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll,mm;
	res(ii/2,mm/2) = ci(ii/2,jj/2) * cj(jj/2,kk/2) * cl(kk/2,ll/2) * ck(ll/2,mm/2); //TODO check if this is the right order !!!!
	return res;
}

//xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim){
//	auto res = xerus::TTOperator(std::vector<size_t>(2*dim,2));
//	if (a!=b && c!=d)
//		res += return_two_e_ac(a,b,d,c,dim);
//		res += return_two_e_ac(a+1,b+1,d+1,c+1,dim);
//	if (c != d)
//		res += return_two_e_ac(a+1,b,d,c+1,dim);
//	res += return_two_e_ac(a,b+1,d+1,c,dim);
//
//
//	return res;
//}



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
