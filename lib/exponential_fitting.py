import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

def show_elbow(y, best_qubit = 5, min_features = 2, show=True, reject_below = 0.99):
    x = np.arange(min_features,min_features+len(y),1)
    y_log = np.log2(y)

    score = []
    first_qubit_number = []

    for start in range(len(x)-1):
        y_fit = y_log[start:]
        x_fit = x[start:]
        model = np.polyfit(x_fit, y_fit, 1)
        
        predict = np.poly1d(model)
        score.append(r2_score(y_fit, predict(x_fit)))
        first_qubit_number.append(x[start])
    
    plt.plot(first_qubit_number,score)
    plt.title('Elbow method - Which n to take for fit?')
    plt.ylabel(r'$R^2$ score')
    plt.xlabel(r'First $n$ number')
    plt.axvline(x = best_qubit+1, c = 'black', linestyle = 'dashed')
    if(show): plt.show()

    return(best_qubit, score)

def show_elbow_n(y, best_qubit = 5, min_features = 2, show=True, trim = 3, rejection_threshold = 0.99):
    y_log = np.log2(y)

    score = []
    first_qubit_number = []

    ns = np.arange(min_features,min_features+len(y),1)
    ns_trimmed = ns[:-trim] # trimm last 3 values
    for n in ns_trimmed:
        start = np.where(ns == n)[0][0]
        y_fit = y_log[start:]
        x_fit = ns[start:]

        model = np.polyfit(x_fit, y_fit, 1)

        predict = np.poly1d(model)
        score.append(r2_score(y_fit, predict(x_fit)))
        first_qubit_number.append(n)

    plt.plot(first_qubit_number, score)
    plt.axvline(x = best_qubit, c = 'black', linestyle = '--')
    plt.show()

    best_index = np.where(np.array(first_qubit_number)==best_qubit)[0][0]
    best_qubit_score = score[best_index]
    print('Best qubit score: ', best_qubit_score)
    if(best_qubit_score < rejection_threshold):
        print('Fit NOT good enough')
    else:
        print('GOOD fit')

    return(first_qubit_number, score)

def show_all_elbows(ys, best_qubit = 5, min_features = 2, show=False):
    for y in ys:
        show_elbow(y, best_qubit = best_qubit, min_features = min_features, show=show)
    plt.show()
    return best_qubit


def fit_exponential_n(y, best_qubit = 5, min_features=2, show_plot = True):
    ns = np.arange(min_features,min_features+len(y),1)
    y_log = np.log2(y)

    best_index = np.where(np.array(ns) == best_qubit)[0][0]

    y_fit = y_log[best_index:]
    x_fit = ns[best_index:]
    model = np.polyfit(x_fit, y_fit, 1)
    alpha = model[0]
    C = np.power(2,model[1])

    if(show_plot):
        plt.plot(ns,y, color = 'black')
        plt.plot(ns, C*np.power(2.0,alpha*ns), color = 'black', linestyle = 'dashed', alpha = 0.5)
        plt.text(3,y[-2],r'C = {:.2f}, $\alpha$ = {:.2f}'.format(C,alpha))
        plt.text(3,y[-1],r'$y = C \cdot 2^{\alpha n}$')
        plt.yscale('log')
        plt.title('Exponential concentration of standard deviations')
        plt.xlabel(r'$n$')
        plt.ylabel(r'$\sigma\left[k(x,x\')\right]$')
        plt.show()
        
    return(alpha, C)