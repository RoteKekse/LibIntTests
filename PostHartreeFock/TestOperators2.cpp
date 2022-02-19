

#include <xerus.h>

using namespace xerus;



int main() {

	size_t d = 4; // 16 electron, 8 electron pairs

	auto hamiltonian = xerus::TTOperator(std::vector<size_t>(2*d,2)); // TT operator initialized with 0

	std::string name = "Data/hamiltonian"+std::to_string(d)+".ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,hamiltonian,xerus::misc::FileFormat::BINARY);
	read.close();

	xerus::Tensor hamiltonian_tensor;
	xerus::Tensor sol(std::vector<size_t>(d,2));
	xerus::Index ii,jj,kk,ll,mm,nn,oo,pp;
	for (size_t i = 0; i< d; ++i){
		auto tmp = hamiltonian.get_component(i);

		if (i == 0){
			hamiltonian_tensor = tmp;
			continue;
		}
		xerus::Tensor hamil_tmp(std::vector<size_t>(2+2*(i+1),2));

		hamil_tmp(ii^(2+2*i-1),kk^3) = hamiltonian_tensor(ii&1,jj) * tmp(jj,kk&1);
		hamiltonian_tensor = hamil_tmp;
	}
	std::cout << "The  hamiltonian_tensor \n" << hamiltonian_tensor << std::endl;

	xerus::Tensor dummy({1});
	dummy[0] = 1.0;
	xerus::Tensor hamiltonian_tensor3(std::vector<size_t>(2*d,2));
	xerus::Tensor hamiltonian_tensor4(std::vector<size_t>(2*d,2));
	hamiltonian_tensor3(ii^(2*d)) = dummy(jj) * hamiltonian_tensor(jj,ii^(2*d),kk) * dummy(kk);
	hamiltonian_tensor4(ii,kk,mm,oo,jj,ll,nn,pp) = hamiltonian_tensor3(ii,jj,kk,ll,mm,nn,oo,pp);

	for (size_t i = 0; i < d - 1; i++){
		std::cout << "The " << i << "th rank of H is " << hamiltonian.ranks()[i] << std::endl;
	}

	auto hamiltonian2 = xerus::TTOperator(hamiltonian_tensor4); // TT operator initialized with 0

	auto lambda = xerus::get_smallest_eigenvalue(sol, hamiltonian_tensor4);

	std::cout << "Lambda =  " << lambda << std::endl;
	std::cout << "Hamil Diff =  " << (hamiltonian - hamiltonian2).frob_norm() << std::endl;
	std::cout << "sol = " << sol << std::endl;


	return 0;
}

