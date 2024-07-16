import unittest
import pandas as pd
import tempfile
from transaction_activity import aggregate_transactions_by_acc, identify_high_risk_transactions, cumulative_balances, monthly_transactions, average_transaction_per_customer, customer_churn
import os

def create_temp_parquet(dataframe):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    dataframe.to_parquet(temp_file.name, index=False)
    return temp_file.name

class TestAggregations(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.accounts_df = pd.DataFrame({
            'account_id': [1, 2, 3, 4],
            'customer_id': [101, 102, 103, 104],
            'account_type': ['checking', 'savings', 'checking', 'savings'],
            'balance': [500, 1000, 1500, 2000],
        })
        
        self.customers_df = pd.DataFrame({
            'customer_id': [101, 102, 103, 104],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com']
        })
        
        self.transactions_df_1 = pd.DataFrame({
            'transaction_id': [1001, 1002, 1003, 1004],
            'account_id': [1, 1, 2, 2],
            'amount': [100, 200, 150, 250],
            'transaction_type': ['debit', 'credit', 'debit', 'credit']
        })
        
        self.transactions_df_2 = pd.DataFrame({
            'transaction_id': [1005, 1006, 1007, 1008],
            'account_id': [3, 3, 4, 4],
            'amount': [300, 400, 500, 600],
            'transaction_type': ['debit', 'credit', 'debit', 'credit']
        })
        
        # Create temporary parquet files
        self.accounts_file = create_temp_parquet(self.accounts_df)
        self.customers_file = create_temp_parquet(self.customers_df)
        self.transactions_file_1 = create_temp_parquet(self.transactions_df_1)
        self.transactions_file_2 = create_temp_parquet(self.transactions_df_2)
        self.transactions_file = create_temp_parquet(pd.concat([self.transactions_df_1, self.transactions_df_2]))

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.accounts_file)
        os.remove(self.customers_file)
        os.remove(self.transactions_file_1)
        os.remove(self.transactions_file_2)
        os.remove(self.transactions_file)
    
    def test_aggregate_transactions_by_acc(self):
        aggregated_data = aggregate_transactions_by_acc(self.transactions_file_1, self.transactions_file_2, self.customers_file)
        self.assertFalse(aggregated_data.empty)
        self.assertIn('customer_id', aggregated_data.columns)
        self.assertIn('account_type', aggregated_data.columns)
        self.assertIn('transaction_type', aggregated_data.columns)
        self.assertIn('amount', aggregated_data.columns)

    def test_identify_high_risk_transactions(self):
        high_risk_data = identify_high_risk_transactions(pd.concat([self.transactions_df_1, self.transactions_df_2]), amount=200)
        self.assertFalse(high_risk_data.empty)
        self.assertIn('account_id', high_risk_data.columns)
        self.assertIn('amount', high_risk_data.columns)

    def test_cumulative_balances(self):
        cumulative_balances_dict = cumulative_balances(self.accounts_file, self.transactions_file)
        self.assertTrue(isinstance(cumulative_balances_dict, dict))
        self.assertIn(1, cumulative_balances_dict)
        self.assertAlmostEqual(cumulative_balances_dict[1], 300)

    def test_monthly_transactions(self):
        # Add 'date' column to transactions data for testing
        self.transactions_df_1['date'] = ['2024-01-01', '2024-01-05', '2024-02-01', '2024-02-05']
        self.transactions_df_2['date'] = ['2024-03-01', '2024-03-05', '2024-04-01', '2024-04-05']
        transactions_file = create_temp_parquet(pd.concat([self.transactions_df_1, self.transactions_df_2]))

        monthly_data = monthly_transactions(transactions_file)
        self.assertFalse(monthly_data.empty)
        self.assertIn('Month', monthly_data.columns)
        self.assertIn('Total Amount', monthly_data.columns)
        self.assertIn('Average Amount', monthly_data.columns)
        self.assertIn('Transaction Count', monthly_data.columns)

        os.remove(transactions_file)

    def test_average_transaction_per_customer(self):
        avg_transaction_data = average_transaction_per_customer(self.transactions_file, self.customers_file)
        self.assertFalse(avg_transaction_data.empty)
        self.assertIn('Customer ID', avg_transaction_data.columns)
        self.assertIn('Average Transaction Amount', avg_transaction_data.columns)

    def test_customer_churn(self):
        churn_model = customer_churn(self.transactions_file, self.accounts_file, self.customers_file, threshold_days_list=30)
        self.assertTrue(hasattr(churn_model, 'predict'))

