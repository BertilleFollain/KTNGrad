from Methods.KRR import KRR
from Methods.KTNGrad import KTNGrad
from Methods.PyMave import PyMave
from Methods.MARS import MARS
from sklearn.utils.estimator_checks import check_estimator

for estimator in [KRR(), KTNGrad(), PyMave(), MARS()]:
    for est, check in check_estimator(estimator, generate_only=True):
        print(str(check))
        try:
            check(est)
        except AssertionError as e:
            print('Failed: ', check, e)
    print(str(estimator) + ' Passed the Scikit-learn Estimator tests')
