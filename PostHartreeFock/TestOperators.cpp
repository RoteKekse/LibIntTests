#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
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
xerus::TTOperator return_one_e_ac2(size_t i, size_t j, size_t d);
xerus::TTOperator return_one_e_ac3(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac2(size_t i, size_t j, size_t k, size_t l, size_t d);

int main() {
	size_t d = 10;
	auto test = xerus::TTTensor(std::vector<size_t>(d,2));

	for(size_t i = 0; i < 5; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[1] = 1.0;
		test.set_component(i,tmp);
	}
	for(size_t i = 5; i < d; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[0] = 1.0;
		test.set_component(i,tmp);
	}


	auto particle_number = xerus::TTOperator(std::vector<size_t>(2*d,2)); // TT operator initialized with 0
	for (size_t i = 0; i < d; i++){
			value_t val = 1.0;
			particle_number += val * return_one_e_ac3(i,i,d);
	}


	xerus::Tensor exp_pn;
	xerus::TTTensor exp_ac1,exp_ac2,exp_ac3;
	xerus::Index ii,jj,kk,ll;
	exp_pn() = particle_number(ii/2,jj/2) * test(ii&0) * test(jj&0);
	XERUS_LOG(Testing, "(P*test,test) =  " << exp_pn[0]);

	for (size_t i = 0; i < d ; ++i){
		auto tmp1 = return_annil(i,d);
		exp_ac1(kk&0) = tmp1(kk/2,ll/2) * test(ll&0);
		for (size_t j = 0; j < d; ++j){
			XERUS_LOG(Testing, "component = " << j << " i = " << i << " (test,test)_op =  " << exp_ac1.get_component(j));
		}
	}
	for (size_t i = 0; i < d ; ++i){
		auto tmp2 = return_create(i,d);
		exp_ac2(kk&0) = tmp2(kk/2,ll/2) * test(ll&0);
		for (size_t j = 0; j < d; ++j){
			XERUS_LOG(Testing, "component = " << j << " i = " << i << " (test,test)_op =  " << exp_ac2.get_component(j));
		}
	}
	for (size_t i = 0; i < d ; ++i){
		auto tmp1 = return_annil(i,d);
		auto tmp2 = return_create(i,d);
		exp_ac3(kk&0) = tmp2(kk/2,ii/2) * tmp1(ii/2,ll/2)  * test(ll&0);
		for (size_t j = 0; j < d; ++j){
			XERUS_LOG(Testing, "component = " << j << " i = " << i << " (test,test)_op =  " << exp_ac3.get_component(j));
		}
	}
	return 0;
}
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
	auto ac_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({1,2,2,1});
	id[{0,0,0,0}] = 1.0;
	id[{0,1,1,0}] = 1.0;
	auto s = xerus::Tensor({1,2,2,1});
	s[{0,0,0,0}] = 1.0;
	s[{0,1,1,0}] = -1.0;
	auto create = xerus::Tensor({1,2,2,1});
	auto annhil = xerus::Tensor({1,2,2,1});
	create[{0,1,0,0}] = 1.0;
	annhil[{0,0,1,0}] = 1.0;

	auto create_annhil = xerus::Tensor({1,2,2,1});
	create_annhil[{0,1,1,0}] = 1.0;

	for (size_t k = 0; k < d; ++k){
		if( (k < i && k < j) || (k > i && k > j)){
			ac_op.set_component(k, id);
		} else if ( (k > i && k < j) || (k < i && k > j)){
			ac_op.set_component(k, s);
		} else if ( k == i && k == j){
			ac_op.set_component(k, create_annhil);
		} else if ( k == i && k != j){
			ac_op.set_component(k, create);
		} else if ( k != i && k == j){
			ac_op.set_component(k, annhil);
		}
	}


	return ac_op;
}

xerus::TTOperator return_one_e_ac2(size_t i, size_t j, size_t d){ // TODO write tests for this


	xerus::Index i1,i2,jj, kk, ll;
	auto ac_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto create = xerus::Tensor({2,2});
	auto annhil = xerus::Tensor({2,2});
	create[{1,0}] = 1.0;
	annhil[{0,1}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp_j = m < j ? s : (m == j ? annhil : id );
		auto tmp_i = m < i ? s : (m == i ? create : id );
		auto tmp = xerus::Tensor({2,2});
		tmp(i1,i2) = tmp_i(i1,jj) * tmp_j(jj,i2);

		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];

		ac_op.set_component(m, res);

	}

	return ac_op;
}

xerus::TTOperator return_one_e_ac3(size_t i, size_t j, size_t d){ // TODO write tests for this
	auto cr = return_create(i,d);
	auto an = return_annil(j,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk;
	res(ii/2,jj/2) = cr(ii/2,kk/2) * an(kk/2, jj/2);
	return res;
}


xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test

	xerus::Index i1,i2,jj, kk, ll;
	auto aacc_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto create = xerus::Tensor({2,2});
	auto annhil = xerus::Tensor({2,2});
	create[{1,0}] = 1.0;
	annhil[{0,1}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp_l = m < l ? s : (m == l ? annhil : id );
		auto tmp_k = m < k ? s : (m == k ? annhil : id );
		auto tmp_j = m < j ? s : (m == j ? create : id );
		auto tmp_i = m < i ? s : (m == i ? create : id );
		auto tmp = xerus::Tensor({2,2});
		tmp(i1,i2) = tmp_i(i1,jj) * tmp_j(jj,kk) * tmp_k(kk,ll) * tmp_l(ll,i2);

		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];

		aacc_op.set_component(m, res);

	}

	return aacc_op;
}


xerus::TTOperator return_two_e_ac2(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test
	auto cr1 = return_create(i,d);
	auto cr2 = return_create(j,d);
	auto an1 = return_annil(k,d);
	auto an2 = return_annil(l,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll,mm;
	res(ii/2,mm/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2);
	return res;
}



