import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pymongo import MongoClient

class FinancialAnalyzer:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['expense-tracker']
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        upi_data = list(self.db.upi_transactions.find({}))
        upi_df = pd.DataFrame(upi_data)
        if not upi_df.empty:
            upi_df['source'] = 'UPI'
            upi_df['type'] = 'Expense'

        cash_data = list(self.db.cash_transactions.find({}))
        cash_df = pd.DataFrame(cash_data)
        if not cash_df.empty:
            cash_df['source'] = 'Cash'
            cash_df['type'] = 'Expense'

        savings_data = list(self.db.daily_savings.find({}))
        savings_df = pd.DataFrame(savings_data)
        if not savings_df.empty:
            savings_df['type'] = 'Savings'

        df = pd.concat([upi_df, cash_df, savings_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_processed_data(self):
        df = self.fetch_data()
        df.set_index('date', inplace=True)

        expenses = df[df['type'] == 'Expense']['amount']
        expenses = expenses.resample('D').sum().fillna(0)

        savings = df[df['type'] == 'Savings']['amount']
        savings = savings.resample('D').sum().fillna(0)

        return expenses, savings

    def plot_expenses(self, expenses, view='W'):
        plt.figure(figsize=(10, 6))
        if view == 'W':
            data = expenses.resample('W').sum()
            title = 'Weekly Expenses'
        else:
            data = expenses.resample('ME').sum()
            title = 'Monthly Expenses'

        plt.plot(data.index, data.values, 'r-', label='Expenses')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Amount (₹)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_savings(self, savings, view='W'):
        plt.figure(figsize=(10, 6))
        if view == 'W':
            data = savings.resample('W').sum()
            title = 'Weekly Savings'
        else:
            data = savings.resample('ME').sum()
            title = 'Monthly Savings'

        plt.plot(data.index, data.values, 'g-', label='Savings')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Amount (₹)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_savings_projection(self, savings, months=3):
        monthly_savings = savings.resample('ME').sum()
        if len(monthly_savings) >= 2:
            plt.figure(figsize=(12, 6))

            plt.plot(monthly_savings.index, monthly_savings.values,
                     label='Historical', color='blue')

            avg_monthly_saving = monthly_savings.mean()
            last_date = monthly_savings.index[-1]
            future_dates = pd.date_range(start=last_date, periods=months+1, freq='ME')[1:]
            trend_line = [monthly_savings.iloc[-1]]

            for _ in range(len(future_dates)):
                trend_line.append(trend_line[-1] + avg_monthly_saving)

            plt.plot(future_dates, trend_line[1:],
                     label='Projection', color='green', linestyle='--')

            plt.title(f'Savings Projection (Next {months} Months)')
            plt.xlabel('Date')
            plt.ylabel('Amount (₹)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            plt.annotate(f'Avg Monthly Saving: ₹{avg_monthly_saving:,.2f}',
                         xy=(0.02, 0.95), xycoords='axes fraction')

            plt.tight_layout()
            plt.show()

            print("\nSavings Insights:")
            print(f"Average Monthly Saving: ₹{avg_monthly_saving:,.2f}")
            print(f"Projected {months}-Month Saving: ₹{avg_monthly_saving * months:,.2f}")

    def predicted_next_month_savings(self, savings):
        monthly_savings = savings.resample('ME').sum()
        if len(monthly_savings) < 2:
            print("Not enough data for prediction.")
            return
        avg_saving = monthly_savings.mean()
        print(f"\nBy the next month, you can save: ₹{avg_saving:,.2f}")

    def predicted_next_week_savings(self, savings):
        weekly_savings = savings.resample('W').sum()
        if len(weekly_savings) < 2:
            print("Not enough data for prediction.")
            return
        avg_weekly_saving = weekly_savings.mean()
        print(f"\nBy the next week, you can save: ₹{avg_weekly_saving:,.2f}")

    def suggested_category_budget(self):
        df = self.fetch_data()
        df.set_index('date', inplace=True)
        expense_df = df[df['type'] == 'Expense']

        if 'category' not in expense_df.columns:
            print("No category-wise data available.")
            return

        last_date = expense_df.index.max()
        three_months_ago = last_date - pd.DateOffset(months=3)
        recent_expenses = expense_df.loc[expense_df.index >= three_months_ago]

        category_summary = recent_expenses.groupby('category')['amount'].mean()

        print("\nSuggested Monthly Budget per Category:")
        for category, amount in category_summary.items():
            print(f"{category}: ₹{amount:,.2f}")

def display_menu():
    print("\n=== Financial Analysis Menu ===")
    print("1. View Expenses (Weekly)")
    print("2. View Expenses (Monthly)")
    print("3. View Savings (Weekly)")
    print("4. View Savings (Monthly)")
    print("5. View Savings Projection")
    print("6. View All Graphs")
    print("7. Predict Next Month Estimated Savings")
    print("8. Suggest Categorywise Budget")
    print("9. Predict Next Week Estimated Savings")
    print("10. Exit")
    return input("Choose an option (1-10): ")

def main():
    analyzer = FinancialAnalyzer()
    expenses, savings = analyzer.get_processed_data()

    while True:
        choice = display_menu()

        if choice == '1':
            analyzer.plot_expenses(expenses, 'W')
        elif choice == '2':
            analyzer.plot_expenses(expenses, 'ME')
        elif choice == '3':
            analyzer.plot_savings(savings, 'W')
        elif choice == '4':
            analyzer.plot_savings(savings, 'ME')
        elif choice == '5':
            months = int(input("Enter number of months for projection (1-12): "))
            months = max(1, min(12, months))
            analyzer.plot_savings_projection(savings, months)
        elif choice == '6':
            analyzer.plot_expenses(expenses, 'W')
            analyzer.plot_expenses(expenses, 'ME')
            analyzer.plot_savings(savings, 'W')
            analyzer.plot_savings(savings, 'ME')
            analyzer.plot_savings_projection(savings)
        elif choice == '7':
            analyzer.predicted_next_month_savings(savings)
        elif choice == '8':
            analyzer.suggested_category_budget()
        elif choice == '9':
            analyzer.predicted_next_week_savings(savings)
        elif choice == '10':
            print("Thank you for using Financial Analyzer!Come back againn")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from pymongo import MongoClient

# class FinancialAnalyzer:
#     def __init__(self):
#         self.client = MongoClient('mongodb://localhost:27017/')
#         self.db = self.client['finance_tracker']
#         self.scaler = MinMaxScaler()

#     def fetch_data(self):
#         # Fetch UPI transactions
#         upi_data = list(self.db.upi_transactions.find({}))
#         upi_df = pd.DataFrame(upi_data)
#         if not upi_df.empty:
#             upi_df['source'] = 'UPI'
#             upi_df['type'] = 'Expense'

#         # Fetch Cash transactions
#         cash_data = list(self.db.cash_transactions.find({}))
#         cash_df = pd.DataFrame(cash_data)
#         if not cash_df.empty:
#             cash_df['source'] = 'Cash'
#             cash_df['type'] = 'Expense'

#         # Fetch Daily savings
#         savings_data = list(self.db.daily_savings.find({}))
#         savings_df = pd.DataFrame(savings_data)
#         if not savings_df.empty:
#             savings_df['type'] = 'Savings'

#         # Combine all data
#         df = pd.concat([upi_df, cash_df, savings_df], ignore_index=True)
#         df['date'] = pd.to_datetime(df['date'])
#         return df

#     def get_processed_data(self):
#         df = self.fetch_data()
#         df.set_index('date', inplace=True)

#         # Process expenses
#         expenses = df[df['type'] == 'Expense']['amount']
#         expenses = expenses.resample('D').sum().fillna(0)

#         # Process savings
#         savings = df[df['type'] == 'Savings']['amount']
#         savings = savings.resample('D').sum().fillna(0)

#         return expenses, savings

#     def plot_expenses(self, expenses, view='W'):
#         plt.figure(figsize=(10, 6))
#         if view == 'W':
#             data = expenses.resample('W').sum()
#             title = 'Weekly Expenses'
#         else:
#             data = expenses.resample('M').sum()
#             title = 'Monthly Expenses'

#         plt.plot(data.index, data.values, 'r-', label='Expenses')
#         plt.title(title)
#         plt.xlabel('Date')
#         plt.ylabel('Amount (₹)')
#         plt.grid(True)
#         plt.xticks(rotation=45)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     def plot_savings(self, savings, view='W'):
#         plt.figure(figsize=(10, 6))
#         if view == 'W':
#             data = savings.resample('W').sum()
#             title = 'Weekly Savings'
#         else:
#             data = savings.resample('M').sum()
#             title = 'Monthly Savings'

#         plt.plot(data.index, data.values, 'g-', label='Savings')
#         plt.title(title)
#         plt.xlabel('Date')
#         plt.ylabel('Amount (₹)')
#         plt.grid(True)
#         plt.xticks(rotation=45)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     def plot_savings_projection(self, savings, months=3):
#         monthly_savings = savings.resample('M').sum()
#         if len(monthly_savings) >= 2:
#             plt.figure(figsize=(12, 6))

#             plt.plot(monthly_savings.index, monthly_savings.values,
#                      label='Historical', color='blue')

#             avg_monthly_saving = monthly_savings.mean()
#             last_date = monthly_savings.index[-1]
#             future_dates = pd.date_range(start=last_date, periods=months+1, freq='M')[1:]
#             trend_line = [monthly_savings.iloc[-1]]

#             for _ in range(len(future_dates)):
#                 trend_line.append(trend_line[-1] + avg_monthly_saving)

#             plt.plot(future_dates, trend_line[1:],
#                      label='Projection', color='green', linestyle='--')

#             plt.title(f'Savings Projection (Next {months} Months)')
#             plt.xlabel('Date')
#             plt.ylabel('Amount (₹)')
#             plt.legend()
#             plt.grid(True)
#             plt.xticks(rotation=45)

#             plt.annotate(f'Avg Monthly Saving: ₹{avg_monthly_saving:,.2f}',
#                          xy=(0.02, 0.95), xycoords='axes fraction')

#             plt.tight_layout()
#             plt.show()

#             print("\nSavings Insights:")
#             print(f"Average Monthly Saving: ₹{avg_monthly_saving:,.2f}")
#             print(f"Projected {months}-Month Saving: ₹{avg_monthly_saving * months:,.2f}")

#     def predicted_next_period_savings(self, savings):
#         monthly_savings = savings.resample('M').sum()
#         if len(monthly_savings) < 2:
#             print("Not enough data for prediction.")
#             return
#         avg_saving = monthly_savings.mean()
#         print(f"\nPredicted Next Month's Saving: ₹{avg_saving:,.2f}")

#     def suggested_category_budget(self):
#         df = self.fetch_data()
#         df.set_index('date', inplace=True)
#         expense_df = df[df['type'] == 'Expense']

#         if 'category' not in expense_df.columns:
#             print("No category-wise data available.")
#             return

#         recent_expenses = expense_df.last('3M')
#         category_summary = recent_expenses.groupby('category')['amount'].mean()

#         print("\nSuggested Monthly Budget per Category:")
#         for category, amount in category_summary.items():
#             print(f"{category}: ₹{amount:,.2f}")

# def display_menu():
#     print("\n=== Financial Analysis Menu ===")
#     print("1. View Expenses (Weekly)")
#     print("2. View Expenses (Monthly)")
#     print("3. View Savings (Weekly)")
#     print("4. View Savings (Monthly)")
#     print("5. View Savings Projection")
#     print("6. View All Graphs")
#     print("7. Predict Next Period Saving")
#     print("8. Suggest Categorywise Budget")
#     print("9. Exit")
#     return input("Choose an option (1-9): ")

# def main():
#     analyzer = FinancialAnalyzer()
#     expenses, savings = analyzer.get_processed_data()

#     while True:
#         choice = display_menu()

#         if choice == '1':
#             analyzer.plot_expenses(expenses, 'W')
#         elif choice == '2':
#             analyzer.plot_expenses(expenses, 'M')
#         elif choice == '3':
#             analyzer.plot_savings(savings, 'W')
#         elif choice == '4':
#             analyzer.plot_savings(savings, 'M')
#         elif choice == '5':
#             months = int(input("Enter number of months for projection (1-12): "))
#             months = max(1, min(12, months))
#             analyzer.plot_savings_projection(savings, months)
#         elif choice == '6':
#             analyzer.plot_expenses(expenses, 'W')
#             analyzer.plot_expenses(expenses, 'M')
#             analyzer.plot_savings(savings, 'W')
#             analyzer.plot_savings(savings, 'M')
#             analyzer.plot_savings_projection(savings)
#         elif choice == '7':
#             analyzer.predicted_next_period_savings(savings)
#         elif choice == '8':
#             analyzer.suggested_category_budget()
#         elif choice == '9':
#             print("Thank you for using Financial Analyzer!")
#             break
#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()

