import sys
import importlib
import argparse
import numpy as np
import traceback


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()

Tensor = mymodule.Tensor

def test_case():

    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[3.0, 2.0], [1.0, 5.0]]), requires_grad=False)
    c = Tensor(np.array([[3.2, 4.5], [6.1, 4.2]]))
    z = np.array([[0.0, 0.0], [0.0, 0.0]])
    sans = a+b
    sans2 = a+a
    mulans = a@b
    mulans2 = (a+b)@c
    sgrad = np.array([[1.0, 1.0], [1.0, 1.0]])
    sgrad2 = np.array([[2.0, 2.0], [2.0, 2.0]])
    mulgrad = np.array([[5.0, 6.0], [5.0, 6.0]])
    mulgrad2 = np.array([[4.0, 4.0], [6.0, 6.0]])
    mulgrad3 = np.array([[7.7, 10.29], [7.7, 10.29]])
    mulgrad4 = np.array([[8.0, 8.0], [13.0, 13.0]])

    try:
        sans.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad, decimal=2)
        print("Test Case 1 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 1 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 2 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 2 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    a.zero_grad()
    b.zero_grad()

    try:
        sans2.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad2, decimal=2)
        print("Test Case 3 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 3 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    a.zero_grad()
    b.zero_grad()

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad, decimal=2)
        print("Test Case 4 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 4 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 5 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 5 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    a.zero_grad()
    b.zero_grad()
    b.requires_grad = True

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(b.grad, mulgrad2, decimal=2)
        print("Test Case 6 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 6 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    a.zero_grad()
    b.zero_grad()
    c.zero_grad()

    try:
        mulans2.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad3, decimal=2)
        np.testing.assert_array_almost_equal(b.grad, mulgrad3, decimal=2)
        print("Test Case 7 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 7 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    try:
        np.testing.assert_array_almost_equal(c.grad, mulgrad4, decimal=2)
        print("Test Case 8 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 8 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()



    try:
        a = Tensor(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]))
        b = Tensor(np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]))
        c = Tensor(np.array([[-27.0, -19.0], [-40.0, -27.0]]))

        a.zero_grad()
        b.zero_grad()
        c.zero_grad()

        mul = (a@b)+c
        mul.backward()

        np.testing.assert_array_almost_equal(a.grad, np.array([[11.,  7.,  3.,], [11.,  7.,  3.,]]), decimal=2)
        np.testing.assert_array_almost_equal(b.grad, np.array([[ 3.,  3.,], [ 7.,  7.,], [11., 11.,]]), decimal=2)
        np.testing.assert_array_almost_equal(c.grad, np.array([[1,1], [1,1]]), decimal=2)
        print("Test Case 9 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 9 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()
    
    try:
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        b = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))

        a.zero_grad()
        b.zero_grad()

        c = a+b
        mul = b@c
        print("\033[92m*************************\033[0m")
        mul.backward()

        # np.testing.assert_array_almost_equal(a.grad, np.array([[1, 1], [1, 1]]), decimal=2)
        np.testing.assert_array_almost_equal(b.grad, np.array([[3, 3], [3, 3]]), decimal=2)
        print("Test Case 10 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 10 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    try:
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        b = Tensor(np.array([[2.0, 0.0], [0.0, 1.0]]))

        a.zero_grad()
        b.zero_grad()

        c = a+b
        mul = c@b
        mul.backward()

        np.testing.assert_array_almost_equal(a.grad, np.array([[2., 1.,], [2., 1.,]]), decimal=2)
        np.testing.assert_array_almost_equal(b.grad, np.array([[5, 4], [4, 3]]), decimal=2)
        print("Test Case 11 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 11 \033[91mFAILED\033[0m", e)
        #traceback.print_exc()

    try:
        a = Tensor(np.array([[56., 58.],[1., 0.],[6., 64.]]))
        b = Tensor(np.array([[52., 51.],[1., 0.]]))

        a.zero_grad()
        b.zero_grad()

        mul = a @ b
        mul.backward()

        np.testing.assert_array_almost_equal(a.grad, np.array([[103.,   1.,],
 [103.,   1.,],
 [103.,   1.,]]), decimal=2)
        np.testing.assert_array_almost_equal(b.grad, np.array([[ 63.,  63.,],
 [122., 122.,]]), decimal=2)
        print("Test Case 12 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 12 \033[91mFAILED\033[0m", e)
        traceback.print_exc()
    
    try:
        a = Tensor(np.array([[56., 58.],[1., 0.],[6., 64.]]))
        b = Tensor(np.array([[52., 51.],[1., 0.]]))

        a.zero_grad()
        b.zero_grad()

        mul = a @ b @ b
        mul.backward()

        np.testing.assert_array_almost_equal(a.grad, np.array([[5407.,  103.,],
 [5407.,  103.,],
 [5407.,  103.,]]), decimal=2)
        np.testing.assert_array_almost_equal(b.grad, np.array([[ 9887.,  3461.,],
 [15779.,  3335.,]]), decimal=2)
        print("Test Case 13 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 13 \033[91mFAILED\033[0m", e)
        traceback.print_exc()
        
    try:
        a = Tensor(np.array([[56., 58., 1.],[1., 0., 5.],[6., 64., 65]]))
        
        a.zero_grad()

        mul = a @ a @ a
        
        mul.backward()
        

        np.testing.assert_array_almost_equal(a.grad, np.array([[18244.,  5244., 22430.,],
 [29151.,  9720., 34517.,],
 [20376.,  6504., 24722.,]]), decimal=2)
        
        a.zero_grad()
        add = a + a + a + a
        add.backward()
        np.testing.assert_array_almost_equal(a.grad, np.array([[4., 4., 4.,],
        [4., 4., 4.,],
        [4., 4., 4.,]]), decimal=2)

        print("Test Case 14 \033[92mPASSED\033[0m")
    except Exception as e:
        print("Test Case 14 \033[91mFAILED\033[0m", e)
        traceback.print_exc()

if __name__ == "__main__": test_case()