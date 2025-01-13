// C++ Program to demonstrate Mathematical model (kuramoto single layer)
#include "Kuramoto.Version4.h" // library Kuramoto version 4 (ubuntu version push in github)

// ====================================================================================================
//       M       M        AA          IIIIII      N     N
//       MMM   MMM       A  A           II        NN    N
//       M  M M  M      AAAAAA          II        N N   N
//       M   M   M     A      A         II        N  N  N
//       M       M    A        A        II        N   N N
//       M       M   A          A     IIIIII      N    NN
// ====================================================================================================
int main() {
    // Hint1: count_rows_cols_file: para in address of file that is ./data.txt
    // Hint2: read_data: first para is number of rows in data file and second para is boolean[0=dont show data,1=show data]
    // data[0]=N & data[1]=L & data[2]=a
    // data[3]=t_0 & data[4]=∆t & data[5]=t_f
    // data[6]=k_0 & data[7]=∆k & data[8]=k_f
    // data[9]=τ_0 & data[10]=∆τ & data[11]=τ_f
    int Num_Sample = 10;
    int EDGES = 20;

    double* data = read_data(count_rows_cols_file("data.txt"), 1);

    std::random_device rd;  // Seed for random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    std::set<std::vector<int>> generated_matrices;

    for (int sample =1; sample < Num_Sample+1; ++sample) {

        double** theta_init = new double*[int(data[0])]; // Allocate memory for an array of 29 pointers to double
        double** w_values = new double*[int(data[0])];   // Allocate memory for an array of 29 pointers to double
        // Allocate memory for the adjacency matrix
        int** adj_layer1 = new int*[int(data[0])];
        for (int i = 0; i < int(data[0]); ++i) {
            adj_layer1[i] = new int[int(data[0])];
        }

        // Generate theta values using uniform distribution
        theta_init = create_uniform_random(theta_init, -M_PI, M_PI, int(data[0]), gen);
        // Generate w values using normal distribution with mean 0 and standard deviation 1
        w_values = create_normal_random(w_values, 0.0, 1.0, int(data[0]), gen);
        // Create the adjacency matrix with exactly 44 edges
        create_adjacency_matrix(adj_layer1, int(data[0]), EDGES, gen, generated_matrices);

        double* frequency_layer1 = new double[int(data[0])];
        double* Phases_initial_layer1 = new double[int(data[0])];

        for (int i = 0; i < int(data[0]); i++) {
            frequency_layer1[i] = (*w_values[i]);
            Phases_initial_layer1[i] = (*theta_init[i]);
        }

        // Free the allocated memory
        for (int i = 0; i < int(data[0]); ++i) {
            delete theta_init[i]; // Delete each allocated double for theta
            delete w_values[i];   // Delete each allocated double for w
        }
        delete[] theta_init; // Delete the array of pointers for theta
        delete[] w_values;   // Delete the array of pointers for w

        // Save the adjacency matrix to a file
        saveas_A(adj_layer1, int(data[0]), sample);
        saveas_W(frequency_layer1, int(data[0]), sample);
        saveas_P(Phases_initial_layer1, int(data[0]), sample);

        double Delay_variable = data[9];
        double* Phases_next_layer1 = new double[int(data[0])];
        double** Phases_history_delay_layer1 = memory_of_delay_of_phases(int(data[0]), Delay_variable, data[4], Phases_initial_layer1);
        double* Phases_layer1_previous = shift_pi2_phases(int(data[0]), Delay_variable, data[4], Phases_history_delay_layer1);

        double Coupling_variable = data[6];
        while (Coupling_variable <= (data[8])) { // Coupling loop
            std::ofstream Save_phases_for_each_coupling("Save/Phase_snapshot_N" + std::to_string(int(data[0])) + "_J" + std::to_string(Coupling_variable) + "_S" + std::to_string(sample) + ".dat");
            double Total_synchrony_layer1 = 0;
            int counter_of_total_sync = 0;
            double Time_variable = data[3]; // reset time for new time
            while (Time_variable < (data[5])) {
                Connected_Constant_Runge_Kutta_4(data, Delay_variable, Coupling_variable, frequency_layer1, adj_layer1, Phases_layer1_previous, Phases_history_delay_layer1, Phases_next_layer1);
                scale_pi(int(data[0]), Phases_layer1_previous); // change values to be in range 0 to 2*Pi
                double synchrony_layer1 = order_parameter(int(data[0]), Phases_layer1_previous); // order parameters

                if (Time_variable >= int(data[5] * 0.981)) { // add sync to total sync
                    for (int i = 0; i < int(data[0]); i++) {
                        Save_phases_for_each_coupling << Phases_layer1_previous[i] << '\t';
                    }
                    Save_phases_for_each_coupling << std::endl;
                    Total_synchrony_layer1 += synchrony_layer1;
                    counter_of_total_sync += 1;

                }
                Time_variable += data[4];
            }
            Total_synchrony_layer1 = Total_synchrony_layer1 / counter_of_total_sync; // calculate total sync and print it
            std::cout << "sample:" << sample << '\t' << "synchrony:" << Total_synchrony_layer1 << std::endl;
            Save_phases_for_each_coupling.close();
            Coupling_variable += data[7]; // next Coupling_variable
        }

        // Free the allocated memory
        for (int i = 0; i < int(data[0]); ++i) {
            delete adj_layer1[i];
        }
        delete[] adj_layer1;

        delete[] frequency_layer1;
        delete[] Phases_initial_layer1;
        delete[] Phases_layer1_previous;
        delete[] Phases_next_layer1;
        delete[] Phases_history_delay_layer1;
    }
    return 0;
}
