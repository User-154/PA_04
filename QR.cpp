#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace std::chrono;

class Vector {
public:
    vector<double> Elem;
    int N;

    Vector(int n) : N(n), Elem(n, 0.0) {}

    Vector Subtract(const Vector& v) const {
        Vector res(N);
        for (int i = 0; i < N; i++) res.Elem[i] = Elem[i] - v.Elem[i];
        return res;
    }

    double Norm() const {
        double sum = 0;
        for (int i = 0; i < N; i++) sum += Elem[i] * Elem[i];
        return sqrt(sum);
    }
};

class Matrix {
public:
    vector<vector<double>> Elem;
    int M, N;

    Matrix(int m, int n) : M(m), N(n), Elem(m, vector<double>(n, 0.0)) {}

    Vector Multiply(const Vector& v) const {
        Vector res(M);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                res.Elem[i] += Elem[i][j] * v.Elem[j];
        return res;
    }

    Vector Multiplication_Trans_Matrix_Vector(const Vector& v) const {
        Vector res(N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                res.Elem[i] += Elem[j][i] * v.Elem[j];
        return res;
    }
};

namespace Substitution_Method {
    void Back_Row_Substitution(const Matrix& R, Vector& X, const Vector& B) {
        for (int i = R.M - 1; i >= 0; i--) {
            if (fabs(R.Elem[i][i]) < 1e-15) throw runtime_error("Zero diagonal element");
            X.Elem[i] = B.Elem[i];
            for (int j = i + 1; j < R.N; j++)
                X.Elem[i] -= R.Elem[i][j] * X.Elem[j];
            X.Elem[i] /= R.Elem[i][i];
        }
    }
}

namespace Gram_Schmidt_Procedure {
    void Classic_Gram_Schmidt_Procedure(const Matrix& A, Matrix& Q, Matrix& R) {
        for (int j = 0; j < A.N; j++) {
            Vector v(A.M);
            for (int i = 0; i < A.M; i++) v.Elem[i] = A.Elem[i][j];

            for (int k = 0; k < j; k++) {
                R.Elem[k][j] = 0;
                for (int i = 0; i < A.M; i++)
                    R.Elem[k][j] += Q.Elem[i][k] * A.Elem[i][j];

                for (int i = 0; i < A.M; i++)
                    v.Elem[i] -= R.Elem[k][j] * Q.Elem[i][k];
            }

            R.Elem[j][j] = v.Norm();
            if (fabs(R.Elem[j][j]) < 1e-15) throw runtime_error("Zero norm in Gram-Schmidt");

            for (int i = 0; i < A.M; i++)
                Q.Elem[i][j] = v.Elem[i] / R.Elem[j][j];
        }
    }
}

class QR_Decomposition {
public:
    Matrix R, Q;

    QR_Decomposition(const Matrix& A) : R(A.M, A.N), Q(A.M, A.M) {
        Gram_Schmidt_Procedure::Classic_Gram_Schmidt_Procedure(A, Q, R);
    }

    Vector Start_Solver(const Vector& F) {
        Vector RES = Q.Multiplication_Trans_Matrix_Vector(F);
        Substitution_Method::Back_Row_Substitution(R, RES, RES);
        return RES;
    }
};

Matrix GenerateMatrix(int N) {
    Matrix A(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A.Elem[i][j] = (i == j) ? 100.0 : 1.0 + 0.3 * (i + 1) - 0.1 * (j + 1);
        }
    }
    return A;
}

int main() {
    setlocale(LC_ALL, "Russian");
    vector<int> sizes = { 250, 500, 1000 };

    for (int N : sizes) {
        cout << "\n=== N = " << N << " ===" << endl;

        Matrix A = GenerateMatrix(N);
        Vector b(N);
        for (int i = 0; i < N; i++) b.Elem[i] = 1.0;

        auto start = high_resolution_clock::now();
        QR_Decomposition QR(A);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "QR-разложение: " << duration.count() << " мс" << endl;

        start = high_resolution_clock::now();
        Vector x = QR.Start_Solver(b);
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << "Решение СЛАУ: " << duration.count() << " мс" << endl;

        Vector Ax = A.Multiply(x);
        Vector residual = Ax.Subtract(b);
        double relative_error = residual.Norm() / b.Norm();
        cout << "Относительная погрешность: " << scientific << relative_error << endl;
    }

    return 0;
}
