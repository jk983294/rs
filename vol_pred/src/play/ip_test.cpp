#include <printemps.h>

int main() {
    /**
        (P1):  minimize       x_1 + 10 x_2
             x
         subject to   66 x_1 + 14 x_2 >= 1430,
                     -82 x_1 + 28 x_2 >= 1306,
                     x_1 and x_2 are integer.
     */
    // (1) Modeling
    printemps::model::IPModel model;

    auto& x = model.create_variables("x", 2, -10000, 10000);
    auto& g = model.create_constraints("g", 2);

    g(0) = 66 * x(0) + 14 * x(1) >= 1430;
    g(1) = -82 * x(0) + 28 * x(1) >= 1306;
    model.minimize(x(0) + 10 * x(1));

    // (2) Running Solver
    auto result = printemps::solver::solve(&model);

    // (3) Accessing the Result
    std::cout << "objective = " << result.solution.objective() << std::endl;
    std::cout << "x(0) = " << result.solution.variables("x").values(0) << std::endl;
    std::cout << "x(1) = " << result.solution.variables("x").values(1) << std::endl;
    return 0;
}