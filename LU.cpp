#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;
using namespace std::chrono;

struct Vector {
    int N;
    vector<double> Elem;

    Vector(int n) : N(n), Elem(n) {}

    double Norm() const {
        double norm = 0;
        for (double v : Elem) norm += v * v;
        return sqrt(norm);
    }

    Vector Subtract(const Vector& other) const {
        Vector result(N);
        for (int i = 0; i < N; i++)
            result.Elem[i] = Elem[i] - other.Elem[i];
        return result;
    }
};

struct Matrix {
    int M;
    vector<vector<double>> Elem;

    Matrix(int m) : M(m), Elem(m, vector<double>(m)) {}

    static Matrix Generate(int m) {
        Matrix mat(m);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(1.0, 2.0);

        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                mat.Elem[i][j] = dis(gen);
                mat.Elem[j][i] = mat.Elem[i][j];
            }
            mat.Elem[i][i] += m;
        }

        return mat;
    }

    Vector Multiply(const Vector& v) const {
        Vector res(M);
        for (int i = 0; i < M; i++) {
            res.Elem[i] = 0;
            for (int j = 0; j < M; j++)
                res.Elem[i] += Elem[i][j] * v.Elem[j];
        }
        return res;
    }
};

struct LU_Decomposition {
    Matrix LU;
    vector<int> P;

    LU_Decomposition(const Matrix& A) : LU(A), P(A.M) {
        int N = A.M;
        for (int i = 0; i < N; i++) P[i] = i;

        for (int i = 0; i < N; i++) {
            // Partial pivoting
            int max_row = i;
            for (int k = i + 1; k < N; k++) {
                if (abs(LU.Elem[k][i]) > abs(LU.Elem[max_row][i]))
                    max_row = k;
            }
            swap(LU.Elem[i], LU.Elem[max_row]);
            swap(P[i], P[max_row]);

            for (int j = i + 1; j < N; j++) {
                double factor = LU.Elem[j][i] / LU.Elem[i][i];
                LU.Elem[j][i] = factor;
                for (int k = i + 1; k < N; k++) {
                    LU.Elem[j][k] -= factor * LU.Elem[i][k];
                }
            }
        }
    }

    void Direct_Way(const Vector& F, Vector& RES) const {
        int N = LU.M;
        for (int i = 0; i < N; i++) {
            RES.Elem[i] = F.Elem[P[i]];
            for (int j = 0; j < i; j++) {
                RES.Elem[i] -= LU.Elem[i][j] * RES.Elem[j];
            }
        }
    }

    void Reverse_Way(const Vector& F, Vector& RES) const {
        int N = LU.M;
        for (int i = N - 1; i >= 0; i--) {
            RES.Elem[i] = F.Elem[i];
            for (int j = i + 1; j < N; j++) {
                RES.Elem[i] -= LU.Elem[i][j] * RES.Elem[j];
            }
            RES.Elem[i] /= LU.Elem[i][i];
        }
    }

    Vector Start_Solver(const Vector& b) const {
        Vector y(LU.M), x(LU.M);
        Direct_Way(b, y);
        Reverse_Way(y, x);
        return x;
    }
};

int main() {
    setlocale(LC_ALL, "Ru");
    vector<int> sizes = { 250, 500, 1000 };

    for (int N : sizes) {
        cout << "\n=== N = " << N << " ===" << endl;

        Matrix A = Matrix::Generate(N);
        Vector b(N);
        for (int i = 0; i < N; i++) b.Elem[i] = 1.0;

        auto start = high_resolution_clock::now();
        LU_Decomposition LU(A);
        auto stop = high_resolution_clock::now();
        auto duration_sec = duration_cast<duration<double>>(stop - start);
        cout << "LU-разложение: " << fixed << setprecision(6) << duration_sec.count() << " сек." << endl;

        start = high_resolution_clock::now();
        Vector x = LU.Start_Solver(b);
        stop = high_resolution_clock::now();
        duration_sec = duration_cast<duration<double>>(stop - start);
        cout << "Решение СЛАУ: " << fixed << setprecision(6) << duration_sec.count() << " сек." << endl;

        Vector Ax = A.Multiply(x);
        Vector residual = Ax.Subtract(b);
        double relative_error = residual.Norm() / b.Norm();
        cout << "Относительная погрешность: " << scientific << setprecision(3) << relative_error << endl;
    }

    return 0;
}
