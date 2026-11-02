# LP-III (410246) Practical Solutions

Every file starts with a detailed **THEORY** comment block covering the related syllabus units, followed by the working code.

Each Python practical also comes as a Jupyter notebook (`.ipynb`, same name) with the theory as formatted text cells and the code split into steps. The Kaggle datasets are included in `ML/`, so the ML code runs as-is.

| No. | File | Topic / Syllabus unit |
|---|---|---|
| DAA1 | `DAA/DAA1_fibonacci_step_count.py` | Step count, iterative vs recursive, asymptotic notation (Units I, II) |
| DAA2 | `DAA/DAA2_job_sequencing_greedy.py` | Greedy method, job sequencing (Unit III) |
| DAA3 | `DAA/DAA3_fractional_knapsack_greedy.py` | Greedy fractional knapsack (Unit III) |
| DAA4 | `DAA/DAA4_01_knapsack_dp_bnb.py` | 0/1 knapsack with DP **and** LC Branch & Bound (Units III, IV) |
| DAA5 | `DAA/DAA5_binomial_coefficient_dp.py` | DP binomial coefficients, Pascal's triangle (Unit III) |
| DAA6 | `DAA/DAA6_8_queens_backtracking.py` | Backtracking N-Queens with first queen fixed (Unit IV) |
| ML1 | `ML/ML1_uber_fare_prediction.py` | Pre-processing, outliers, correlation, LR vs Random Forest |
| ML2 | `ML/ML2_email_spam_knn_svm.py` | KNN vs SVM classification |
| ML3 | `ML/ML3_bank_churn_neural_network.py` | ANN / MLP churn classifier |
| ML4 | `ML/ML4_gradient_descent.py` | Gradient descent on y=(x+3)² |
| ML5 | `ML/ML5_knn_diabetes.py` | KNN with confusion matrix, accuracy, error rate, precision, recall |
| ML6 | `ML/ML6_kmeans_hierarchical_sales.py` | K-Means + hierarchical clustering, elbow method |
| BCT2 | `BCT/BCT2_Metamask_Wallet.md` | MetaMask wallet procedure + theory |
| BCT3 | `BCT/BCT3_Bank.sol` | Bank smart contract (deposit / withdraw / balance) |
| BCT4 | `BCT/BCT4_StudentData.sol` | Struct, array, fallback; gas observation |

## Running

**DAA:** run `python3 DAA/DAA1_fibonacci_step_count.py`. Each program asks for input, and pressing Enter uses built-in sample data.

**ML:** the datasets are already in `ML/`: `uber.csv`, `emails.csv`, `Churn_Modelling.csv`, `diabetes.csv`, `sales_data_sample.csv`.
Run, for example, `python3 ML/ML1_uber_fare_prediction.py`, or open the matching `.ipynb` in Jupyter and run all cells. Keep each CSV in the same folder as the code that reads it.
Required libraries: `pip install numpy pandas scikit-learn matplotlib seaborn scipy`.
ML3 uses TensorFlow/Keras if it is installed (`pip install tensorflow`); otherwise it uses scikit-learn's MLPClassifier.

**BCT:** open https://remix.ethereum.org, paste the `.sol` file, compile with 0.8.x and deploy on Remix VM or Sepolia via MetaMask. Steps are in each file's header.
