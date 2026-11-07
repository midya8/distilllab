import unittest

from distilllab.eval import run_evaluation


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_evaluation()

    def test_teacher_is_strong(self):
        self.assertGreaterEqual(self.r["teacher_acc"], 0.80)

    def test_distillation_gain_is_real(self):
        self.assertGreaterEqual(self.r["distill_gain"], 0.04)

    def test_control_stays_at_chance(self):
        # mismatched soft targets carry no input-specific information:
        # if the control ever learns, the dark-knowledge isolation fails.
        self.assertLessEqual(self.r["control_acc"], 0.20)

    def test_t1_ablation_direction(self):
        # T=1 keeps the teacher's argmax supervision but sheds similarity
        # information; on this fixture it lands no higher than T=4.
        self.assertLessEqual(self.r["ablation_t1_acc"], self.r["distilled_acc"])

    def test_distilled_matches_teacher_more_closely(self):
        self.assertLess(self.r["kl_distilled"], self.r["kl_direct"])

    def test_distillation_reduces_overfit_gap(self):
        self.assertGreaterEqual(self.r["direct_train_acc"], 0.99)  # direct memorizes
        self.assertLess(self.r["overfit_gap_distilled"], self.r["overfit_gap_direct"])

    def test_student_is_much_smaller(self):
        self.assertGreaterEqual(self.r["teacher_params"] / self.r["student_params"], 10)

    def test_determinism(self):
        self.assertEqual(run_evaluation(), self.r)


if __name__ == "__main__":
    unittest.main()
