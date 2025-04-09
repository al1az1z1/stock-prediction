#!/usr/bin/env python
# coding: utf-8

# In[16]:


def predict_all_companies(multi_features, single_features, 
                          multi_class_model, multi_reg_model, 
                          single_class_model, single_reg_model,
                          scaler_m_class, scaler_m_reg,
                          scaler_s_class, scaler_s_reg,
                          multi_stock_data):
    companies = multi_features['symbol'].unique()
    results = []
    
    from datetime import datetime, timedelta
    import pandas as pd
    
    latest_date = pd.to_datetime(multi_features['date'].max())
    
    def next_business_day(date):
        next_day = date + timedelta(days=1)
        if next_day.weekday() >= 5:
            next_day += timedelta(days=7 - next_day.weekday())
        return next_day
    
    prediction_date = next_business_day(latest_date)
    
    for company in companies:
        multi_company_data = multi_features[multi_features['symbol'] == company]
        
        if len(multi_company_data) == 0:
            continue
            
        latest_multi_data = multi_company_data.sort_values('date').iloc[-1]
        latest_date_str = latest_multi_data['date']
        previous_date_str = multi_company_data.sort_values('date').iloc[-2]['date'] if len(multi_company_data) > 1 else "N/A"
        
        single_company_data = single_features[single_features['symbol'] == company] if company in single_features['symbol'].unique() else None
        
        X_multi_class = latest_multi_data.drop(['date', 'symbol', 'next_day_close', 'price_change', 'pct_change', 'target'])
        X_multi_reg = latest_multi_data.drop(['date', 'symbol', 'target', 'price_change', 'pct_change', 'next_day_close'])
        
        X_multi_class_scaled = scaler_m_class.transform(X_multi_class.values.reshape(1, -1))
        X_multi_reg_scaled = scaler_m_reg.transform(X_multi_reg.values.reshape(1, -1))
        
        multi_direction_prob = multi_class_model.predict_proba(X_multi_class_scaled)[0][1]
        multi_direction_pred = 1 if multi_direction_prob > 0.5 else 0
        multi_price_pred = multi_reg_model.predict(X_multi_reg_scaled)[0]
        
        latest_price = latest_multi_data['close']
        
        multi_expected_direction = 1 if multi_price_pred > latest_price else 0
        
        if multi_direction_pred != multi_expected_direction:
            if multi_direction_pred == 1:
                multi_price_pred = latest_price * 1.007
            else:
                multi_price_pred = latest_price * 0.994
        
        multi_price_change = multi_price_pred - latest_price
        multi_pct_change = (multi_price_pred / latest_price - 1) * 100
        multi_confidence = max(multi_direction_prob, 1-multi_direction_prob) * 100
        
        if single_company_data is not None and len(single_company_data) > 0:
            latest_single_data = single_company_data.sort_values('date').iloc[-1]
            
            X_single_class = latest_single_data.drop(['date', 'symbol', 'next_day_close', 'price_change', 'pct_change', 'target'])
            X_single_reg = latest_single_data.drop(['date', 'symbol', 'target', 'price_change', 'pct_change', 'next_day_close'])
            
            X_single_class_scaled = scaler_s_class.transform(X_single_class.values.reshape(1, -1))
            X_single_reg_scaled = scaler_s_reg.transform(X_single_reg.values.reshape(1, -1))
            
            single_direction_prob = single_class_model.predict_proba(X_single_class_scaled)[0][1]
            single_direction_pred = 1 if single_direction_prob > 0.5 else 0
            single_price_pred = single_reg_model.predict(X_single_reg_scaled)[0]
            
            single_expected_direction = 1 if single_price_pred > latest_price else 0
            if single_direction_pred != single_expected_direction:
                if single_direction_pred == 1:
                    single_price_pred = latest_price * 1.007
                else:
                    single_price_pred = latest_price * 0.994
                    
            single_price_change = single_price_pred - latest_price
            single_pct_change = (single_price_pred / latest_price - 1) * 100
            single_confidence = max(single_direction_prob, 1-single_direction_prob) * 100
            
            better_model = "Multi-Stock" if multi_confidence > single_confidence else "Single-Stock"
        else:
            single_direction_pred = None
            single_price_pred = None
            single_price_change = None
            single_pct_change = None
            single_confidence = None
            better_model = "Multi-Stock"
        
        results.append({
            'Symbol': company,
            'Previous Date': previous_date_str,
            'Latest Date': latest_date_str,
            'Next Prediction Date': prediction_date.strftime('%Y-%m-%d'),
            'Latest Price': latest_price,
            'Multi-Model Direction': 'UP ↑' if multi_direction_pred == 1 else 'DOWN ↓',
            'Multi-Model Price': multi_price_pred,
            'Multi-Model Change': multi_price_change,
            'Multi-Model Change %': multi_pct_change,
            'Multi-Model Confidence': multi_confidence,
            'Single-Model Direction': 'UP ↑' if single_direction_pred == 1 else 'DOWN ↓' if single_direction_pred is not None else "N/A",
            'Single-Model Price': single_price_pred,
            'Single-Model Change': single_price_change,
            'Single-Model Change %': single_pct_change,
            'Single-Model Confidence': single_confidence,
            'Recommended Model': better_model
        })
    
    predictions_df = pd.DataFrame(results)
    predictions_df = predictions_df.sort_values('Multi-Model Change %', ascending=False)
    
    predictions_df['Latest Price'] = predictions_df['Latest Price'].round(2)
    predictions_df['Multi-Model Price'] = predictions_df['Multi-Model Price'].round(2)
    predictions_df['Multi-Model Change'] = predictions_df['Multi-Model Change'].round(2)
    predictions_df['Multi-Model Change %'] = predictions_df['Multi-Model Change %'].round(2)
    predictions_df['Multi-Model Confidence'] = predictions_df['Multi-Model Confidence'].round(1)
    
    numeric_cols = ['Single-Model Price', 'Single-Model Change', 'Single-Model Change %', 'Single-Model Confidence']
    for col in numeric_cols:
        predictions_df[col] = pd.to_numeric(predictions_df[col], errors='coerce').round(2)
    
    def check_agreement(row):
        if row['Single-Model Direction'] == "N/A":
            return "N/A"
        elif row['Multi-Model Direction'] == row['Single-Model Direction']:
            return "Yes"
        else:
            return "No"
            
    predictions_df['Models Agree'] = predictions_df.apply(check_agreement, axis=1)
    predictions_df['High Confidence'] = predictions_df['Multi-Model Confidence'] > 70
    
    return predictions_df

# Add this to the end of your script
print("\n=== PREDICTIONS FOR ALL COMPANIES ===")
all_predictions = predict_all_companies(
    multi_features, single_features,
    multi_class_model, multi_reg_model,
    single_class_model, single_reg_model,
    scaler_m_class, scaler_m_reg,
    scaler_s_class, scaler_s_reg,
    multi_stock_data
)

# Print top gainers
print(f"\nTop 5 stocks with highest predicted gains:")
top_5_columns = ['Symbol', 'Latest Date', 'Next Prediction Date', 'Latest Price', 
                'Multi-Model Direction', 'Multi-Model Price', 'Multi-Model Change %', 
                'Multi-Model Confidence', 'Single-Model Direction', 'Single-Model Change %',
                'Recommended Model']
print(all_predictions[top_5_columns].head(5).to_string(index=False))

# Print biggest losers
print(f"\nBottom 5 stocks with largest predicted losses:")
print(all_predictions[top_5_columns].tail(5).to_string(index=False))

# Model comparison table
model_comp = all_predictions.dropna(subset=['Single-Model Change %']).copy()
model_comp['Difference'] = model_comp['Multi-Model Change %'] - model_comp['Single-Model Change %']
model_comp['Abs Difference'] = abs(model_comp['Difference'])

print("\n=== MODEL COMPARISON SUMMARY ===")
comparison_cols = ['Symbol', 'Latest Price', 'Multi-Model Change %', 
                  'Single-Model Change %', 'Difference', 'Models Agree', 'Recommended Model']
print(model_comp.sort_values('Abs Difference', ascending=False)[comparison_cols].head(10).to_string(index=False))

# High confidence picks
high_conf = all_predictions[all_predictions['High Confidence'] == True]
print(f"\nHigh confidence predictions (>70%):")
print(high_conf[['Symbol', 'Next Prediction Date', 'Latest Price', 
                'Multi-Model Direction', 'Multi-Model Change %', 
                'Multi-Model Confidence', 'Models Agree']].head(5).to_string(index=False))

# Model agreement stats
agree_count = (all_predictions['Models Agree'] == 'Yes').sum()
disagree_count = (all_predictions['Models Agree'] == 'No').sum()
na_count = (all_predictions['Models Agree'] == 'N/A').sum()
total_valid = agree_count + disagree_count

print(f"\nModel Agreement Analysis:")
print(f"Models agree on {agree_count} stocks ({agree_count/total_valid*100:.1f}% of comparable predictions)")
print(f"Models disagree on {disagree_count} stocks ({disagree_count/total_valid*100:.1f}% of comparable predictions)")
print(f"Single-stock model not available for {na_count} stocks")


report_lines = []
report_lines.append("\n==================================================")
report_lines.append("         STOCK PREDICTION MODEL COMPARISON         ")
report_lines.append("==================================================\n")

# Overall comparison based on metrics
report_lines.append("DIRECTION PREDICTION PERFORMANCE SUMMARY:")
report_lines.append("-----------------------------------------")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
    single_val = class_metrics[class_metrics['Metric'] == metric]['Single Stock'].values[0]
    multi_val = class_metrics[class_metrics['Metric'] == metric]['Multi-Stock'].values[0]
    diff = class_metrics[class_metrics['Metric'] == metric]['Difference'].values[0]
    pct = class_metrics[class_metrics['Metric'] == metric]['% Change'].values[0]
    better = "BETTER" if diff > 0 else "WORSE"
    report_lines.append(f"{metric}: Single-Stock: {single_val:.4f}, Multi-Stock: {multi_val:.4f}")
    report_lines.append(f"   Difference: {diff:.4f} ({pct:.2f}%) - Multi-Stock model is {better}\n")

report_lines.append("PRICE PREDICTION PERFORMANCE SUMMARY:")
report_lines.append("------------------------------------")
for metric in ['MSE', 'RMSE', 'MAE', 'R² Score']:
    single_val = reg_metrics[reg_metrics['Metric'] == metric]['Single Stock'].values[0]
    multi_val = reg_metrics[reg_metrics['Metric'] == metric]['Multi-Stock'].values[0]
    diff = reg_metrics[reg_metrics['Metric'] == metric]['Difference'].values[0]
    pct = reg_metrics[reg_metrics['Metric'] == metric]['% Change'].values[0]
    if metric in ['MSE', 'RMSE', 'MAE']:
        better = "BETTER" if diff < 0 else "WORSE"
    else:
        better = "BETTER" if diff > 0 else "WORSE"
    report_lines.append(f"{metric}: Single-Stock: {single_val:.4f}, Multi-Stock: {multi_val:.4f}")
    report_lines.append(f"   Difference: {diff:.4f} ({pct:.2f}%) - Multi-Stock model is {better}\n")

# Stock-specific analysis
report_lines.append("STOCK-SPECIFIC MODEL COMPARISON:")
report_lines.append("--------------------------------")
model_diff = all_predictions.dropna(subset=['Single-Model Change %']).copy()
model_diff['Abs Difference'] = abs(model_diff['Multi-Model Change %'] - model_diff['Single-Model Change %'])

# Top 5 stocks with largest model differences
top_diff = model_diff.nlargest(5, 'Abs Difference')
report_lines.append("Top 5 Stocks with Largest Prediction Differences:")
for _, row in top_diff.iterrows():
    report_lines.append(f"- {row['Symbol']}: Multi-Stock predicts {row['Multi-Model Change %']:.2f}%, "
                      f"Single-Stock predicts {row['Single-Model Change %']:.2f}%")
    report_lines.append(f"  Absolute Difference: {row['Abs Difference']:.2f}%, Models Agree: {row['Models Agree']}")

# Model agreement summary
report_lines.append(f"\nOverall Model Agreement: {agree_count}/{total_valid} stocks ({agree_count/total_valid*100:.1f}%)")
report_lines.append(f"Overall Model Disagreement: {disagree_count}/{total_valid} stocks ({disagree_count/total_valid*100:.1f}%)")

# Final recommendation
better_model_counts = model_diff['Recommended Model'].value_counts()
multi_count = better_model_counts.get('Multi-Stock', 0)
single_count = better_model_counts.get('Single-Stock', 0)
total_comp = multi_count + single_count

report_lines.append("\nMODEL RECOMMENDATION SUMMARY:")
report_lines.append("----------------------------")
report_lines.append(f"Multi-Stock model recommended for {multi_count}/{total_comp} stocks ({multi_count/total_comp*100:.1f}%)")
report_lines.append(f"Single-Stock model recommended for {single_count}/{total_comp} stocks ({single_count/total_comp*100:.1f}%)")

if multi_count > single_count:
    report_lines.append("\nOVERALL CONCLUSION: Multi-Stock approach appears to be more effective for most stocks.")
    if agree_count > disagree_count:
        report_lines.append("The models generally agree on prediction direction, but Multi-Stock model shows higher confidence.")
    else:
        report_lines.append("However, there is significant disagreement between models, suggesting company-specific factors may be important.")
else:
    report_lines.append("\nOVERALL CONCLUSION: Single-Stock approach appears to be more effective for most stocks.")
    report_lines.append("This suggests that individual stock characteristics may be more important than market-wide patterns.")

# Print the report
print("\n\n")
for line in report_lines:
    print(line)

# Save the report to a file
with open('model_comparison_report.txt', 'w') as f:
    for line in report_lines:
        f.write(line + '\n')

# Save all data
all_predictions.to_csv('all_company_predictions.csv', index=False)

# Create visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Top stocks chart
plt.figure(figsize=(12, 8))
top_gainers = all_predictions.head(10)
sns.barplot(x='Multi-Model Change %', y='Symbol', data=top_gainers, palette='viridis')
plt.title('Top 10 Stocks by Predicted Percentage Gain')
plt.xlabel('Predicted Change (%)')
plt.tight_layout()
plt.savefig('top_stock_predictions.png', dpi=300)

# Model comparison
plt.figure(figsize=(14, 8))
comparable = all_predictions.dropna(subset=['Single-Model Change %']).copy()
comparable['Absolute Difference'] = abs(comparable['Multi-Model Change %'] - comparable['Single-Model Change %'])
top_diff = comparable.nlargest(15, 'Absolute Difference')

plt.subplot(1, 2, 1)
model_comparison = pd.melt(top_diff[['Symbol', 'Multi-Model Change %', 'Single-Model Change %']], 
                          id_vars=['Symbol'], var_name='Model', value_name='Change %')
sns.barplot(x='Change %', y='Symbol', hue='Model', data=model_comparison)
plt.title('Stocks with Largest Prediction Differences')
plt.xlabel('Predicted Change (%)')
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.scatterplot(x='Multi-Model Change %', y='Single-Model Change %', data=comparable)
plt.axline([0, 0], [1, 1], color='red', linestyle='--')
plt.title('Model Prediction Correlation')
plt.xlabel('Multi-Stock Model Change %')
plt.ylabel('Single-Stock Model Change %')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('model_comparison.png', dpi=300)

# Calendar heatmap 
colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)

if len(all_predictions) > 30:
    selected_stocks = all_predictions.sample(min(30, len(all_predictions)))
else:
    selected_stocks = all_predictions

plt.figure(figsize=(12, 10))
pivot_data = selected_stocks.pivot(index='Symbol', columns='Next Prediction Date', values='Multi-Model Change %')
sns.heatmap(pivot_data, cmap=cmap, center=0, annot=True, fmt='.1f', linewidths=.5)
plt.title('Predicted Change % by Stock and Date')
plt.tight_layout()
plt.savefig('prediction_calendar.png', dpi=300)


# In[ ]:




