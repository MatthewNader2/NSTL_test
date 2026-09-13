"""
Tests for Cross-Domain Bridge Cells and Unification.
Validates:
1. 5 Pillow <-> OpenCV bridge cells (channel inversion [:, :, ::-1], color_bgr, color_rgb, gray).
2. Pandas to_numpy bridges (PANDAS_DATAFRAME_TO_NUMPY, PANDAS_SERIES_TO_NUMPY).
3. Sklearn accepted_states for shared vocabulary (ndarray_generic, dataframe_2d_generic).
4. Seaborn axes_with_content reuse from Matplotlib.
"""

import unittest
from lattice import LatticeOrchestrator, TypeRegistry, AlgebraicSignature


class TestCrossDomainBridges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = LatticeOrchestrator()

    def test_pillow_to_cv2_bgr_order(self):
        """Pillow RGB to OpenCV BGR bridge unifies with CV2_IMWRITE."""
        pil_bgr = self.orchestrator.loaded_cells.get("PILLOW_RGB_TO_NDARRAY_BGR_ORDER")
        cv2_write = self.orchestrator.loaded_cells.get("CV2_IMWRITE")
        self.assertIsNotNone(pil_bgr)
        self.assertIsNotNone(cv2_write)

        # Check code template has channel inversion
        self.assertIn("[:, :, ::-1]", pil_bgr.code_template)
        # Check output state
        out_sig = pil_bgr.outputs["output_var"].signature
        self.assertEqual(out_sig.state, "color_bgr")
        self.assertEqual(out_sig.type_name, "ndarray")

        # Unifies with CV2_IMWRITE image port
        cv2_in = cv2_write.inputs["image"]
        self.assertTrue(out_sig.unifies_with(cv2_in))

    def test_pillow_to_cv2_grayscale(self):
        """Pillow grayscale to OpenCV grayscale bridge unifies with CV2_CANNY."""
        pil_gray = self.orchestrator.loaded_cells.get("PILLOW_GRAYSCALE_TO_NDARRAY")
        cv2_canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        self.assertIsNotNone(pil_gray)
        self.assertIsNotNone(cv2_canny)

        out_sig = pil_gray.outputs["output_var"].signature
        self.assertEqual(out_sig.state, "gray")

        # CV2_CANNY expects src in state gray
        self.assertTrue(out_sig.unifies_with(cv2_canny.inputs["src"]))

    def test_cv2_to_pillow_bgr(self):
        """OpenCV BGR array unifies with PILLOW_FROM_NDARRAY_BGR."""
        cv2_read = self.orchestrator.loaded_cells.get("CV2_IMREAD")
        pil_from_bgr = self.orchestrator.loaded_cells.get("PILLOW_FROM_NDARRAY_BGR")
        self.assertIsNotNone(cv2_read)
        self.assertIsNotNone(pil_from_bgr)

        # Inversion in template
        self.assertIn("[:, :, ::-1]", pil_from_bgr.code_template)

        cv2_out = cv2_read.outputs["output_var"].signature
        pil_in = pil_from_bgr.inputs["array"]
        self.assertTrue(cv2_out.unifies_with(pil_in))

    def test_pandas_to_sklearn_bridges(self):
        """Pandas to_numpy bridges unify with Sklearn train_test_split."""
        pd_df = self.orchestrator.loaded_cells.get("PANDAS_DATAFRAME_TO_NUMPY")
        pd_s = self.orchestrator.loaded_cells.get("PANDAS_SERIES_TO_NUMPY")
        sk_split = self.orchestrator.loaded_cells.get("sklearn.model_selection.train_test_split")
        self.assertIsNotNone(pd_df)
        self.assertIsNotNone(pd_s)
        self.assertIsNotNone(sk_split)

        # DataFrame to NumPy -> Sklearn X
        self.assertTrue(pd_df.primary_output.signature.unifies_with(sk_split.inputs["X"]))
        # Series to NumPy -> Sklearn y
        self.assertTrue(pd_s.primary_output.signature.unifies_with(sk_split.inputs["y"]))

    def test_sklearn_shared_vocabulary(self):
        """Sklearn nodes declare accepted_states for ndarray_generic / dataframe_2d_generic."""
        sk_split = self.orchestrator.loaded_cells.get("sklearn.model_selection.train_test_split")
        x_sig = sk_split.inputs["X"].signature
        self.assertIn("ndarray_generic", x_sig.accepted_states)
        self.assertIn("dataframe_2d_generic", x_sig.accepted_states)

    def test_seaborn_axes_reuse_from_matplotlib(self):
        """Seaborn plotting nodes accept Matplotlib Axes in axes_initialized state."""
        mpl_sub = self.orchestrator.loaded_cells.get("matplotlib_plt_subplots_single")
        sns_hist = self.orchestrator.loaded_cells.get("sns_histplot_univariate")
        self.assertIsNotNone(mpl_sub)
        self.assertIsNotNone(sns_hist)

        mpl_ax = mpl_sub.outputs["ax"].signature
        sns_ax = sns_hist.inputs["port_4"].signature

        self.assertEqual(mpl_ax.type_name, "Axes")
        self.assertEqual(mpl_ax.state, "axes_initialized")
        self.assertIn("axes_initialized", sns_ax.accepted_states)
        self.assertTrue(mpl_ax.unifies_with(sns_ax))


if __name__ == "__main__":
    unittest.main()
