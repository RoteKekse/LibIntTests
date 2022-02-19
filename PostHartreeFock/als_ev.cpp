
#include <xerus.h>

using namespace xerus;

class InternalSolver {
	const size_t d;
	double lambda;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x)
		: d(_x.degree()), x(_x), A(_A), maxIterations(200), lambda(1.0)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		std::vector<double> residuals(10, 1000.0);
		std::vector<double> residuals_ev(10, 1000.0);

		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			residuals_ev.push_back(lambda);
			if (residuals_ev.back()/residuals_ev[residuals_ev.size()-10] > 0.99) {
				XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
				return lambda; // We are done!
			}
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back() << " Eigenvalue " << lambda);


			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;
				//XERUS_LOG(simpleALS, "corePosition: " << corePosition << " Eigenvalue " << lambda);
				XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back() << " Eigenvalue " << lambda);

				const Tensor &Ai = A.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);

				lambda = xerus::get_smallest_eigenvalue(x.component(corePosition), op);

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}

		}
		return lambda;
	}
};

double simpleALS(const TTOperator& _A, TTTensor& _x)  {
	InternalSolver solver(_A, _x);
	return solver.solve();
}


int main() {
	XERUS_LOG(simpleALS,"Begin Solving for smallest eigenvalue ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");

	size_t d = 48; // 16 electron, 8 electron pairs

	auto hamiltonian = xerus::TTOperator(std::vector<size_t>(2*d,2)); // TT operator initialized with 0
	auto particle_number = xerus::TTOperator(std::vector<size_t>(2*d,2)); // TT operator initialized with 0

	std::string name = "Data/hamiltonian"+std::to_string(d)+".ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,hamiltonian,xerus::misc::FileFormat::BINARY);
	read.close();
	std::string name2 = "Data/particle_number"+std::to_string(d)+".ttoperator";
	std::ifstream read2(name2.c_str());
	misc::stream_reader(read2,particle_number,xerus::misc::FileFormat::BINARY);
	read2.close();

	auto phi = TTTensor(std::vector<size_t>(d,2));
	xerus::Index i,j;
	xerus::Tensor exp_pn;
	xerus::Tensor exp_hamilton;


	for(size_t i = 0; i < 10; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[1] = 1.0;
		phi.set_component(i,tmp);
	}
	for(size_t i = 10; i < d; ++i){
		auto tmp = xerus::Tensor({1,2,1});
		tmp[0] = 1.0;
		phi.set_component(i,tmp);
	}
	exp_pn() = particle_number(i/2,j/2) * phi(i&0) * phi(j&0);
	exp_hamilton() = hamiltonian(i/2,j/2) * phi(i&0) * phi(j&0);
	XERUS_LOG(info, "(P*phi,P) =  " << exp_pn[0]);
	XERUS_LOG(info, "(H*phi,H) =  " << exp_hamilton[0]);

	double lambda = simpleALS(hamiltonian, phi);



	for (size_t i = 0; i < d - 1; i++){
		std::cout << "The " << i << "th rank of H is " << hamiltonian.ranks()[i] << std::endl;
		std::cout << "The " << i << "th rank of phi is " << phi.ranks()[i] << std::endl;
	}


	exp_pn() = particle_number(i/2,j/2) * phi(i&0) * phi(j&0);
	exp_hamilton() = hamiltonian(i/2,j/2) * phi(i&0) * phi(j&0);


	auto hamiltonian_T = hamiltonian;
	hamiltonian_T.transpose();
	XERUS_LOG(info, "E0 =  " << lambda);
	XERUS_LOG(info, "(P*phi,P) =  " << exp_pn[0]);
	XERUS_LOG(info, "(H*phi,H) =  " << exp_hamilton[0]);
	XERUS_LOG(info, "Norm H - H^T =  " << (hamiltonian - hamiltonian_T).frob_norm());

}

