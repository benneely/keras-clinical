import unittest
import numpy as np
from keras_clinical import applications


class GraceMortality(unittest.TestCase):

    def test_single_patient(self):
        gm = applications.GRACE_MORTALITY()
        pretend_patient_1 = np.array([[57, 70, 110, 1.2, 3, 1, 0, 1]], dtype=np.float32)
        pt1 = gm.predict(pretend_patient_1)
        print(pt1)
        self.assertEqual(pt1, np.array([[0.21693133]], dtype=np.float32))

    def test_multiple_patients(self):
        gm = applications.GRACE_MORTALITY()
        pretend_patients =  np.array(
            [[57, 70, 110, 1.2, 3, 1, 0, 1],
             [57, 70, 110, 1.2, 3, 1, 0, 1],
             [57, 70, 110, 1.2, 3, 1, 0, 1],
             [57, 70, 110, 1.2, 3, 1, 0, 1],
             [57, 70, 110, 1.2, 3, 1, 0, 1]],
            dtype=np.float32
        )
        pts = gm.predict(pretend_patients)
        self.assertTrue((pts.tolist() == np.array([[ 0.21693127],
           [ 0.21693127],
           [ 0.21693127],
           [ 0.21693127],
           [ 0.21693127]], dtype=np.float32)).all())