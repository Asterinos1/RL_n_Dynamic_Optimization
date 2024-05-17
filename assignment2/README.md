# Day Trading in the Stock Market

## Assignment 2: Optimizing Stock Investments Using Bandit Algorithms

### Description

In this assignment, your task is to use the adversarial framework of bandit algorithms we learned (i.e., experts, adversarial bandit algorithms, OCO) to optimize your investments in stocks. The setup of the problem is a VERY simplified version of day-to-day trading.

During the morning of each day, you invest exactly 1 euro per day on one of K stocks. At the end of the day, you learn the closing price for this stock and the percentage (%) of increase or decrease of that stock. You then sell your stock, and you profit exactly that percentage of your 1 euro.

For example:
- If you invested in stock 3 on day t, and it closed with a gain of 5%, you get back your 1 euro plus 5 cents.
- A stock can also lose value during the day. You still must sell and pick a stock the next morning. For instance, if stock 3 lost 10%, then you lose 10 cents (but you will still invest a whole euro the next day).

The next day you can invest again 1 euro on any of the K stocks, and so forth, until the end of the horizon.

### Data

In the `stocks.csv` file, you will find the day-to-day percentage changes in 10 real stocks compiled from a global stock exchange, for a duration of 2000 days.

### Tasks

#### Task 1: Experts Setup
Assume that at the end of each day you learn the price change percentage for all K stocks (not just the one you invested in). Implement the Multiplicative Weights algorithm to maximize the amount of profit you have collected at the end of the horizon. You should show two separate plots:
1. Cumulative regret of your algorithm, from day 1 to the last day.
2. Cumulative profits of your algorithm (i.e., how much you've made in total by day 2, day 3, etc.)

#### Task 2: Experts with Transaction Fees
Assume the previous experts setup again (i.e., full feedback) but now each stock n has a fixed transaction cost \(C_i\) that differs between stocks.

For example, imagine you invested again in stock 3, and let's say that stock 3 has a transaction cost \(C_3 = 2%\) and shows a price increase of 5%. Then, at the end of the day, you will have lost 2 cents for the transaction, and gained 5 from the price increase, for an overall gain of 3 cents.

Assume that the transaction fees are: 0.5%, 1%, 1.5%, ..., for stocks 0 to K, respectively.

Modify your MW algorithm to maximize the accumulated profit (gains minus the transaction costs). Show two plots again, the cumulative regret and the cumulative profit over time, but each one together with the respective plot from the previous scenario with no fees.

#### Task 3: Bandits with Transaction Fees
Assume finally that you have a bandit setup, rather than an experts one. That is, at the end of day t you only learn the % increase/decrease of the stock you invested in that day (but not the other ones). Modify your previous algorithm to be applicable in this bandit setup and maximize the accumulated profit (minus transactions costs). Plot the cumulative regret and cumulative profits for this scenario and compare with the respective experts plots.

### Instructions

1. Load the `stocks.csv` data.
2. Implement the Multiplicative Weights algorithm for Task 1.
3. Modify the algorithm to account for transaction fees for Task 2.
4. Adjust the algorithm to a bandit setup for Task 3.
5. Generate the required plots for cumulative regret and cumulative profit for each task.
6. Compare and analyze the results.

### Plots to Generate

1. **Task 1**:
    - Cumulative Regret
    - Cumulative Profit

2. **Task 2**:
    - Cumulative Regret (with and without fees)
    - Cumulative Profit (with and without fees)

3. **Task 3**:
    - Cumulative Regret (bandit with fees, experts with and without fees)
    - Cumulative Profit (bandit with fees, experts with and without fees)
