"""BIZ-05: юнит-экономика — арифметика, окупаемость, точка безубыточности, CLI."""
import unittest
from dataclasses import replace

from invent_core import economics as ec


class TestEconomics(unittest.TestCase):
    def test_hand_calculation(self):
        i = ec.Inputs(cells=1000, counts_per_year=10, manual_sec_per_cell=36, crew_rate_rub_h=1000,
                      truck_rate_rub_h=500, drone_cells_per_hour=500, drone_operator_share=0.5,
                      drone_capex_rub=120000, drone_opex_rub_year=50000)
        r = ec.evaluate(i)
        self.assertAlmostEqual(r.manual_hours_year, 100.0)          # 1000 × 10 × 36 с
        self.assertAlmostEqual(r.manual_cost_year, 150000.0)        # 100 ч × 1500 ₽
        self.assertAlmostEqual(r.drone_hours_year, 20.0)
        self.assertAlmostEqual(r.drone_cost_year, 60000.0)          # 20 × 0.5 × 1000 + 50000
        self.assertAlmostEqual(r.savings_year, 90000.0)
        self.assertAlmostEqual(r.payback_months, 16.0)              # 120000 / 7500
        self.assertAlmostEqual(ec.breakeven_counts(i), 50000 / (15000 - 1000))

    def test_not_profitable_and_subscription(self):
        i = ec.Inputs(counts_per_year=0.1)
        self.assertIsNone(ec.evaluate(i).payback_months)
        r = ec.evaluate(replace(ec.Inputs(), subscription_rub_month=10000))
        self.assertEqual(r.payback_months, 0.0)                     # подписка: вложений клиента нет

    def test_report_mentions_illustration(self):
        text = ec.report(ec.Inputs(), ec.evaluate(ec.Inputs()))
        self.assertIn("иллюстрация", text)
        self.assertIn("20 000 ячеек, 12 пересчётов", text)


if __name__ == "__main__":
    unittest.main()
