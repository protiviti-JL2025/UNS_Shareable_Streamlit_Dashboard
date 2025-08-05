import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import calendar

def download_chart(fig, filename="chart.png", label="📥 Download Chart as PNG"):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")

st.set_page_config(layout="wide")
st.title("Sales Return Deep Dive Dashboard")

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Upload File
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    # Ensure column names are consistent
    df.columns = df.columns.str.strip()

    st.sidebar.header("Select Analysis Type")
    analysis_type = st.sidebar.selectbox("Choose", ['Analysis 1 - Overall View', 'Analysis 2 - Deep Dive into each Return Type', 'Analysis 3 - BRAND-based Deep Dive (Top 5)', 'Analysis 5 - Customers with only ER Returns', 'Analysis 6 - Closing Stock & DSI Trend', 'Analysis 7 - High Sales Return Ratio Customers'])

    return_types = ['ER', 'NE', 'DR', 'SR']
    return_df = df[df['Tran_type'].isin(return_types)]
    return_counts = return_df['Tran_type'].value_counts().reindex(return_types).fillna(0)

    if analysis_type == 'Analysis 1 - Overall View':
        st.subheader("Analysis 1 : Overall Return Type Distribution")

        # Dropdown for metric selection
        metric_option = st.selectbox(
            "Select Metric for Analysis",
            ["All", "Value Counts", "Gross Amount", "Return Quantity"]
        )

        def plot_metric(metric_series, title, ylabel, filename):
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(metric_series.index, metric_series.values, color='skyblue')
            ax.set_title(title)
            ax.set_xlabel('Return Type')
            ax.set_ylabel(ylabel)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', ha='center', va='bottom')

            st.pyplot(fig)
            download_chart(fig, filename)

            # Summary Table
            st.write(f"### Summary Table: % Contribution in Top 4 by {ylabel}")
            summary_table = pd.DataFrame({
                'Return Type': metric_series.index.tolist(),
                f'% in Top 4 Types ({ylabel})': [(count / metric_series.sum()) * 100 for count in metric_series]
            }).set_index('Return Type')

            st.dataframe(summary_table.style.format({
                f'% in Top 4 Types ({ylabel})': "{:.1f}%"
            }))

        valid_types = ['ER', 'DR', 'NE', 'SR']
        filtered_df = df[df['Tran_type'].isin(valid_types)]
        
        if metric_option == "All":
            return_counts = filtered_df['Tran_type'].value_counts()

            plot_metric(return_counts, 'Total No. of Return Transactions by Type (ER, NE, DR, SR)', 'Count', "Overall_Count.png")

            # Gross Amount
            gross_by_type = filtered_df.groupby('Tran_type')['gross_amount'].sum()
            plot_metric(gross_by_type, 'Total Gross Amount by Return Type', 'Gross Amount (₹)', "Overall_Gross.png")

            # Return Quantity
            qty_by_type = return_counts = filtered_df.groupby('Tran_type')['Return_qty'].sum()
            plot_metric(qty_by_type, 'Total Return Quantity by Return Type', 'Return Quantity', "Overall_Qty.png")

        elif metric_option == "Value Counts":
            return_counts = filtered_df['Tran_type'].value_counts()
            plot_metric(return_counts, 'Total No. of Return Transactions by Type (ER, NE, DR, SR)', 'Count', "Overall_Count.png")

        elif metric_option == "Gross Amount":
            gross_by_type = filtered_df.groupby('Tran_type')['gross_amount'].sum()
            plot_metric(gross_by_type, 'Total Gross Amount by Return Type', 'Gross Amount (₹)', "Overall_Gross.png")

        elif metric_option == "Return Quantity":
            qty_by_type = return_counts = filtered_df.groupby('Tran_type')['Return_qty'].sum()
            plot_metric(qty_by_type, 'Total Return Quantity by Return Type', 'Return Quantity', "Overall_Qty.png")

        st.divider()

    elif analysis_type == 'Analysis 2 - Deep Dive into each Return Type':
        st.subheader("Analysis 2 : Deep Dive into each Return Type")
        # Step 1: Metric Selection
        metric_option = st.selectbox("Select Metric", ["Return Quantity", "Gross Amount"])

        # Step 2: Tran_type Selection
        return_types = ['ER', 'DR', 'NE', 'SR']  # allowed types
        selected_type = st.selectbox("Select Return Type to Deep Dive", return_types)

        df_type = df[df['Tran_type'] == selected_type]

        # Choose the metric column
        if metric_option == "Return Quantity":
            metric_column = "Return_qty"
        elif metric_option == "Gross Amount":
            metric_column = "gross_amount"

        # REGION LEVEL
        st.subheader(f"2. Top Regions for Return Type: {selected_type}")

        if metric_column:
            region_group = df_type.groupby("Region")[metric_column].sum()
        else:
            region_group = df_type['Region'].value_counts()

        top_regions = region_group.sort_values(ascending=False).head(10)
        all_regions = region_group

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_regions.index, y=top_regions.values, ax=ax_r, palette='Reds')
        ax_r.set_ylabel(metric_column if metric_column else "Return Count")
        ax_r.set_title(f"Top Regions by {selected_type} - {metric_option}")

        for i, v in enumerate(top_regions.values):
            ax_r.text(i, v + 50, str(int(v)), ha='center', fontweight='bold')

        st.pyplot(fig_r)
        download_chart(fig_r, "region_return_chart.png")

        # Region Summary Table
        region_summary = pd.DataFrame({
            '% Contribution in Top': (top_regions / top_regions.sum() * 100).round(2)
        })
        st.write("### Region Contribution Summary")
        st.dataframe(region_summary)

        # AREA LEVEL
        selected_region = st.selectbox("Select Region to See Area Breakdown", top_regions.index)
        df_region = df_type[df_type['Region'] == selected_region]

        if metric_column:
            area_group = df_region.groupby("Area")[metric_column].sum()
        else:
            area_group = df_region['Area'].value_counts()

        top_areas = area_group.sort_values(ascending=False).head(10)
        all_areas = area_group

        st.subheader(f"3. Top Areas in {selected_region} for {selected_type} - {metric_option}")
        fig_a, ax_a = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_areas.index, y=top_areas.values, ax=ax_a, color='orange')
        ax_a.set_ylabel(metric_column if metric_column else "Return Count")
        ax_a.set_title(f"Top Areas in {selected_region} by {selected_type} - {metric_option}")
        plt.xticks(rotation=45)

        for i, v in enumerate(top_areas.values):
            ax_a.text(i, v + 10, str(int(v)), ha='center')

        st.pyplot(fig_a)
        download_chart(fig_a, "area_return_chart.png")

        # Area Summary Table
        area_summary = pd.DataFrame({
            '% Contribution in Top': (top_areas / top_areas.sum() * 100).round(2)
        })
        st.write("### Area Contribution Summary")
        st.dataframe(area_summary)

        # CUSTOMER LEVEL
        selected_area = st.selectbox("Select Area to See CUST_NAME Breakdown", top_areas.index)
        df_area = df_region[df_region['Area'] == selected_area]
        customer_group = df_area.groupby('CUST_NAME')[['Return_qty', 'gross_amount']].sum()

        if metric_column:
            top_customers = customer_group.sort_values(by=metric_column, ascending=False).head(10)
            all_customers = customer_group.sort_values(by=metric_column, ascending=False)
        else:
            top_customers = df_area['CUST_NAME'].value_counts().head(10)
            all_customers = df_area['CUST_NAME'].value_counts()

        st.subheader(f"4. Top Customers in {selected_area} by {selected_type} - {metric_option}")
        fig_cust, ax_cust = plt.subplots(figsize=(10, 6))
        if metric_column:
            sns.barplot(x=top_customers[metric_column], y=top_customers.index, ax=ax_cust, palette='Blues_d')
            ax_cust.set_xlabel(metric_column)
        else:
            sns.barplot(x=top_customers.values, y=top_customers.index, ax=ax_cust, palette='Blues_d')
            ax_cust.set_xlabel("Return Count")

        ax_cust.set_title(f"Top Customers in {selected_area} by {selected_type} - {metric_option}")
        st.pyplot(fig_cust)
        download_chart(fig_cust, "customer_return_chart.png")

        # Customer Summary Table
        if metric_column:
            customer_summary = pd.DataFrame({
                '% Contribution in Top': (top_customers[metric_column] / top_customers[metric_column].sum() * 100).round(2),
                '% Contribution in All Customers in Area': (top_customers[metric_column] / all_customers[metric_column].sum() * 100).round(2)
            })
        else:
            customer_summary = pd.DataFrame({
                '% Contribution in Top': (top_customers / top_customers.sum() * 100).round(2),
                '% Contribution in All Customers in Area': (top_customers / all_customers.sum() * 100).round(2)
            })
        st.write("### CUST_NAME Contribution Summary")
        st.dataframe(customer_summary)

        # Always show both Return_qty and gross_amount at the end
        st.write("### Return and Gross Amount for Top Customers")
        if isinstance(top_customers, pd.Series):
            top_customer_names = top_customers.index
            top_customers_metrics = customer_group.loc[top_customer_names]
        else:
            top_customers_metrics = top_customers

        st.dataframe(top_customers_metrics.style.format({'Return_qty': '{:,.0f}', 'gross_amount': '₹{:,.0f}'}))

        # Download last graph
        buf = BytesIO()
        fig_cust.savefig(buf, format="png")
        st.download_button("📥 Download CUST_NAME Graph as PNG", data=buf.getvalue(), file_name="top_customers.png", mime="image/png")

    elif analysis_type == 'Analysis 3 - BRAND-based Deep Dive (Top 5)':
        st.subheader("Analysis 3: BRAND-based Deep Dive (Top 5)")

        # --- Metric selection ---
        metric_option = st.selectbox("Select Metric", ['Return Quantity', 'Gross Amount'])
        if metric_option == 'Return Quantity':
            metric_column = 'Return_qty'
        elif metric_option == 'Gross Amount':
            metric_column = 'gross_amount'

        # Filter valid return types
        return_types = ['ER', 'NE', 'DR', 'SR']
        df_returns = df[df['Tran_type'].isin(return_types)]

        # Step 1: Top 5 Brands by selected metric
        top_brands = (
            df_returns.groupby('BRAND')[metric_column]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index
        )
        df_top5 = df_returns[df_returns['BRAND'].isin(top_brands)]

        pivot = df_top5.pivot_table(
            index='BRAND', columns='Tran_type', values=metric_column,
            aggfunc='sum', fill_value=0
        )
        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)
        pivot_chart = pivot.drop(columns='Total')

        # --- Bar chart: Top Brands ---
        st.markdown(f"#### 📦 Top 5 Brands by {metric_option} (ER, DR, SR, NE)")
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_chart[return_types].plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')

        for i, brand in enumerate(pivot_chart.index):
            total = pivot.loc[brand, 'Total']
            y_offset = 0
            for rt in return_types:
                value = pivot_chart.loc[brand, rt]
                # if value > 0: 
                #if the above line is uncommented then the % for gross amount will not show
                # snce all gross amount values are -ive for any return. 
                pct = value / total * 100
                ax.text(i, y_offset + value / 2, f"{pct:.1f}%", ha='center', va='center', fontsize=9)
                y_offset += value
            ax.text(i, y_offset + 10, f"Total: {int(total)}", ha='center', fontweight='bold', fontsize=10)

        ax.set_title(f"Top 5 Brands by {metric_option}", fontsize=14)
        ax.set_ylabel(metric_option)
        ax.set_xlabel("BRAND")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title="Return Type")
        st.pyplot(fig)
        download_chart(fig, "Brand_Return.png")

        # Summary table
        full_brand_metric = df_returns.groupby('BRAND')[metric_column].sum()
        summary = pd.DataFrame({
            metric_option: pivot['Total'],
            '% of Top 5': (pivot['Total'] / pivot['Total'].sum() * 100).round(2),
            '% of All Brands': (pivot['Total'] / full_brand_metric.sum() * 100).round(2)
        })
        st.dataframe(summary.style.format({metric_option: '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Brands': '{:.2f}%'}))

        # --- Region-wise ---
        selected_brand = st.selectbox("Select a BRAND", top_brands)
        brand_df = df_returns[df_returns['BRAND'] == selected_brand]

        region_data = brand_df.groupby(['Region', 'Tran_type'])[metric_column].sum().unstack(fill_value=0)
        region_data['Total'] = region_data.sum(axis=1)
        region_data = region_data.sort_values('Total', ascending=False)
        region_top = region_data.drop(columns='Total').head(5)

        st.markdown(f"#### 🌎 Region-wise {metric_option} for {selected_brand}")
        fig, ax = plt.subplots(figsize=(12, 6))
        region_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (region, row) in enumerate(region_top.iterrows()):
            total = row.sum()
            y_offset = 0
            for rt in return_types:
                value = row.get(rt, 0)
                if value > 0:
                    pct = value / total * 100
                    ax.text(i, y_offset + value / 2, f"{pct:.1f}%", ha='center', va='center', fontsize=8)
                    y_offset += value
            ax.text(i, y_offset + 10, f"{int(total)}", ha='center', fontweight='bold', fontsize=9)
        ax.set_title(f"{selected_brand} - {metric_option} by Region", fontsize=14)
        ax.set_ylabel(metric_option)
        ax.set_xlabel("Region")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Return Type')
        st.pyplot(fig)
        download_chart(fig, "Region_Return.png")

        region_summary = pd.DataFrame({
            metric_option: region_top.sum(axis=1),
            '% of Top 5': (region_top.sum(axis=1) / region_top.sum().sum() * 100).round(2),
            '% of All Regions': (region_top.sum(axis=1) / brand_df.groupby('Region')[metric_column].sum().sum() * 100).round(2)
        })
        st.dataframe(region_summary.style.format({metric_option: '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Regions': '{:.2f}%'}))

        # --- Area-wise ---
        selected_region = st.selectbox("Select a Region", region_top.index)
        area_df = brand_df[brand_df['Region'] == selected_region]

        area_data = area_df.groupby(['Area', 'Tran_type'])[metric_column].sum().unstack(fill_value=0)
        area_data['Total'] = area_data.sum(axis=1)
        area_data = area_data.sort_values('Total', ascending=False)
        area_top = area_data.drop(columns='Total').head(5)

        st.markdown(f"#### 🏙️ Area-wise {metric_option} for {selected_brand} in {selected_region}")
        fig, ax = plt.subplots(figsize=(12, 6))
        area_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (area, row) in enumerate(area_top.iterrows()):
            total = row.sum()
            y_offset = 0
            for rt in return_types:
                value = row.get(rt, 0)
                if value > 0:
                    pct = value / total * 100
                    ax.text(i, y_offset + value / 2, f"{pct:.1f}%", ha='center', va='center', fontsize=8)
                    y_offset += value
            ax.text(i, y_offset + 10, f"{int(total)}", ha='center', fontweight='bold', fontsize=9)
        ax.set_title(f"{selected_brand} - {metric_option} by Area in {selected_region}", fontsize=14)
        ax.set_ylabel(metric_option)
        ax.set_xlabel("Area")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Return Type')
        st.pyplot(fig)
        download_chart(fig, "Area_Return.png")

        area_summary = pd.DataFrame({
            metric_option: area_top.sum(axis=1),
            '% of Top 5': (area_top.sum(axis=1) / area_top.sum().sum() * 100).round(2),
            '% of All Areas': (area_top.sum(axis=1) / area_df.groupby('Area')[metric_column].sum().sum() * 100).round(2)
        })
        st.dataframe(area_summary.style.format({metric_option: '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Areas': '{:.2f}%'}))

        # --- Customer-wise ---
        selected_area = st.selectbox("Select an Area", area_top.index)
        cust_df = area_df[area_df['Area'] == selected_area]

        cust_data = cust_df.groupby(['CUST_NAME', 'Tran_type'])[metric_column].sum().unstack(fill_value=0)
        cust_data['Total'] = cust_data.sum(axis=1)
        cust_data = cust_data.sort_values('Total', ascending=False)
        cust_top = cust_data.drop(columns='Total').head(5)

        st.markdown(f"#### 🧍 CUST_NAME-wise {metric_option} in {selected_area}, {selected_region}")
        fig, ax = plt.subplots(figsize=(12, 6))
        cust_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (cust, row) in enumerate(cust_top.iterrows()):
            total = row.sum()
            y_offset = 0
            for rt in return_types:
                value = row.get(rt, 0)
                if value > 0:
                    pct = value / total * 100
                    ax.text(i, y_offset + value / 2, f"{pct:.1f}%", ha='center', va='center', fontsize=7)
                    y_offset += value
            ax.text(i, y_offset + 5, f"{int(total)}", ha='center', fontweight='bold', fontsize=8)
        ax.set_title(f"{selected_brand} - {metric_option} by CUST_NAME in {selected_area}", fontsize=14)
        ax.set_ylabel(metric_option)
        ax.set_xlabel("CUST_NAME")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Return Type')
        st.pyplot(fig)
        download_chart(fig, "Customer_Return.png")

        cust_summary = pd.DataFrame({
            metric_option: cust_top.sum(axis=1),
            '% of Top 5': (cust_top.sum(axis=1) / cust_top.sum().sum() * 100).round(2),
            '% of All Customers': (cust_top.sum(axis=1) / cust_df.groupby('CUST_NAME')[metric_column].sum().sum() * 100).round(2)
        })
        st.dataframe(cust_summary.style.format({metric_option: '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Customers': '{:.2f}%'}))

        # --- Summary block ---
        selected_customer = st.selectbox("Select a CUST_NAME", cust_top.index)
        cust_summary_df = df[
            (df['CUST_NAME'] == selected_customer) &
            (df['BRAND'] == selected_brand) &
            (df['Region'] == selected_region) &
            (df['Area'] == selected_area) &
            (df['Tran_type'].isin(return_types))
        ]
        #total_sales = cust_summary_df['sale_qty'].sum()
        total_returns = cust_summary_df['Return_qty'].sum()
        total_gross = cust_summary_df['gross_amount'].sum()

        st.markdown(f"### 📊 Summary for {selected_customer}")
        #st.markdown(f"- **Total Sales Quantity:** {int(total_sales)}")
        st.markdown(f"- **Total Return Quantity:** {int(total_returns)}")
        st.markdown(f"- **Total Gross Amount:** ₹{int(total_gross)}")

    elif analysis_type == 'Analysis 5 - Customers with only ER Returns':

        # Step 1: Filter only for return types
        return_types = ['ER', 'NE', 'DR', 'SR']
        df_returns = df[df['Tran_type'].isin(return_types)].copy()

        # Step 2: Group by customer and get unique return types
        customer_return_types = df_returns.groupby('CUST_NAME')['Tran_type'].unique()

        # Step 3: Identify customers who only returned ER
        customers_only_er = customer_return_types[customer_return_types.apply(lambda x: set(x) == {'ER'})].index.tolist()

        # Step 4: Filter the dataframe for these customers
        df_customers_only_er = df_returns[df_returns['CUST_NAME'].isin(customers_only_er)].copy()

        # Step 5: Apply gross_amount > 10,000 filter
        gross_summary = df_customers_only_er.groupby('CUST_NAME')['gross_amount'].sum().reset_index()
        qualified_customers = gross_summary[gross_summary['gross_amount'] < -10000]['CUST_NAME'].tolist()
        df_customers_only_er = df_customers_only_er[df_customers_only_er['CUST_NAME'].isin(qualified_customers)].copy()

        # Step 6: Calculate Sales Ratio
        returns_df = df[df['Tran_type'].isin(['ER', 'SR', 'DR', 'NE'])]
        sales_df = df[df['Tran_type'].isin(['INV', 'IC'])]

        return_summary = returns_df.groupby('CUST_NAME')['Return_qty'].sum().reset_index(name='Total_Returns')
        sales_summary = sales_df.groupby('CUST_NAME')['sale_qty'].sum().reset_index(name='Total_Sales')

        ratio_df = pd.merge(return_summary, sales_summary, on='CUST_NAME', how='inner')
        ratio_df['Sales_Ratio'] = ratio_df['Total_Returns'] / ratio_df['Total_Sales']

        # Step 7: Apply Sales Ratio > 0.03 filter
        high_ratio_customers = ratio_df[ratio_df['Sales_Ratio'] > 0.03]['CUST_NAME'].tolist()

        # Step 8: Final qualified customers (Gross amount > 10K and Sales Ratio > 3%)
        final_customers = list(set(qualified_customers) & set(high_ratio_customers))
        df_customers_only_er = df_customers_only_er[df_customers_only_er['CUST_NAME'].isin(final_customers)].copy()

        # Step 9: Final display
        st.subheader("Customers Who Only Returned ER (Expired Returns)")
        # st.write(f"Total qualifying customers (Gross Amount > 10,000): {len(final_customers)}")

        # Table 1: Detailed view
        st.markdown("#### Detailed Customer Records (Region, Area, Brand Level)")
        st.dataframe(
            df_customers_only_er[['CUST_NAME', 'Region', 'Area', 'BRAND', 'Return_qty', 'gross_amount']]
            .drop_duplicates()
            .sort_values(by='CUST_NAME')
        )

        # Table 2: Aggregated summary with return qty, gross amount, sales ratio
        agg_df = df_customers_only_er.groupby('CUST_NAME').agg({
            'Return_qty': 'sum',
            'gross_amount': 'sum'
        }).reset_index()

        # Add Sales Ratio from ratio_df
        final_summary = pd.merge(agg_df, ratio_df[['CUST_NAME', 'Sales_Ratio']], on='CUST_NAME', how='left')

        st.markdown("#### Summary Table (Customers with Gross Amount > 10000 and Sales ration > 3% i.e. 0.03)")
        st.dataframe(final_summary.sort_values(by='Sales_Ratio', ascending=False))

    elif analysis_type == "Analysis 6 - Closing Stock & DSI Trend":
        st.subheader("Analysis 6: SR/NE Returns vs 3-Month Stock & DSI")

        # --- Metric selector for SR/NE bar chart ---
        metric_option = st.selectbox("Select Metric", ['Return Quantity', 'Gross Amount'])
        if metric_option == 'Return Quantity':
            metric_column = 'Return_qty'
        elif metric_option == 'Gross Amount':
            metric_column = 'gross_amount'

        # --- Upload and load Sheet 2 ---
        uploaded_file_sheet2 = st.file_uploader(
            "Upload Sheet 2 (Closing Stock + DSI)", type=["xlsx"], key="sheet2"
        )
        if not uploaded_file_sheet2:
            st.info("Please upload the second sheet to continue.")
            st.stop()

        sheet1 = df.copy()
        sheet2 = load_data(uploaded_file_sheet2)

        # --- Parse dates & extract month periods ---
        sheet1["INV_DATE"] = pd.to_datetime(sheet1["INV_DATE"], errors="coerce")
        sheet1["Month_Year"] = sheet1["INV_DATE"].dt.to_period("M")

        sheet2["StartDt"] = pd.to_datetime(sheet2["StartDt"], format="%d-%m-%Y", errors="coerce")
        sheet2["Month_Year"] = sheet2["StartDt"].dt.to_period("M")

        # --- Make dropdown of valid months (>= Jul 2024) ---
        months = sorted(sheet2["Month_Year"].dropna().unique())
        valid_months = [m for m in months if m >= pd.Period("2024-07", freq="M")]
        month_labels = [m.strftime("%B %Y") for m in valid_months]
        sel_label = st.selectbox("Select Month:", month_labels)
        sel_period = valid_months[month_labels.index(sel_label)]

        # --- Filter SR+NE returns in selected month & find top 10 customers ---
        returns = sheet1[
            sheet1["Tran_type"].isin(["SR","NE"]) &
            (sheet1["Month_Year"] == sel_period)
        ]
        cust_pv = (
            returns
            .groupby(["CUST_NAME","Tran_type"])[metric_column]
            .sum()
            .unstack(fill_value=0)
        )
        cust_pv["Total"] = cust_pv.sum(axis=1)
        top_cust = cust_pv["Total"].sort_values(ascending=False).head(10).index.tolist()
        plot_df = cust_pv.loc[top_cust, ["SR","NE","Total"]].reset_index()

        # --- Stacked bar chart ---
        fig = go.Figure()
        for rt,color in [("SR","indianred"),("NE","steelblue")]:
            fig.add_trace(go.Bar(
                x=plot_df["CUST_NAME"], y=plot_df[rt],
                name=rt, marker_color=color,
                text=plot_df[rt], textposition="inside"
            ))
        for i,c in enumerate(plot_df["CUST_NAME"]):
            total = plot_df.loc[i,"Total"]
            fig.add_annotation(
                x=c, y=total*1.02,
                text=f"{int(total):,}", showarrow=False, font=dict(size=12)
            )
        fig.update_layout(
            barmode="stack",
            title=f"Top 10 Customers by {metric_option} (SR+NE) in {sel_label}",
            xaxis_title="Customer", yaxis_title=metric_option,
            legend_title="Return Type", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Pivot Sheet 2 by CUST_NAME & Month_Year for ClsStk & DSI ---
        stock_grp = (
            sheet2
            .groupby(["CUST_NAME","Month_Year"], as_index=False)
            .agg(ClsStk_total=("ClsStk","sum"),
                DSI_mean=("DSI","mean"))
        )

        # --- Determine the three months before sel_period ---
        prior = [sel_period - i for i in (1,2,3)]
        labels = [p.strftime("%b-%Y") for p in prior]

        # --- Overall customer table ---
        rows = []
        for cust in top_cust:
            d = {"Customer":cust}
            sub = stock_grp[stock_grp["CUST_NAME"]==cust]
            for lbl,p in zip(labels, prior):
                sf = sub[sub["Month_Year"]==p]
                #d[f"{lbl} ClsStk"] = sf["ClsStk_total"].sum() if not sf.empty else 0
                d[f"{lbl} DSI"]    = sf["DSI_mean"].iloc[0] if not sf.empty else np.nan
            rows.append(d)
        overall_df = pd.DataFrame(rows).set_index("Customer")

        st.markdown("### Top 10 Customers: 3-Month ClsStk & DSI")
        st.dataframe(overall_df.style.format({
            #**{f"{l} ClsStk":"{:,.0f}" for l in labels},
            **{f"{l} DSI":"{:.1f}"   for l in labels}
        }))

        # --- Customer dropdown for brand-level details ---
        chosen = st.selectbox("Select Customer for Brand-level View", top_cust)
        cust_stock = stock_grp[stock_grp["CUST_NAME"]==chosen]

        # --- Top 5 brands by SR+NE for that customer ---
        brand_pv = (
            returns[returns["CUST_NAME"]==chosen]
            .groupby(["BRAND","Tran_type"])[metric_column]
            .sum().unstack(fill_value=0)
        )
        brand_pv["Total"] = brand_pv.sum(axis=1)
        top_brands = brand_pv["Total"].sort_values(ascending=False).head(5).index.tolist()

        # --- Build brand table ---
        brow = []
        for b in top_brands:
            d = {"Brand":b}
            for lbl,p in zip(labels, prior):
                sf = cust_stock[cust_stock["CUST_NAME"]==chosen]
                #d[f"{lbl} ClsStk"] = sf["ClsStk_total"][sf["CUST_NAME"]==chosen][sf["Month_Year"]==p].sum() if not sf.empty else 0
                d[f"{lbl} DSI"]    = sf["DSI_mean"][sf["CUST_NAME"]==chosen][sf["Month_Year"]==p].mean() if not sf.empty else np.nan
            brow.append(d)
        brand_df = pd.DataFrame(brow).set_index("Brand")

        st.markdown(f"### {chosen}: Top 5 Brands ClsStk & DSI")
        st.dataframe(brand_df.style.format({
            #**{f"{l} ClsStk":"{:,.0f}" for l in labels},
            **{f"{l} DSI":"{:.1f}"   for l in labels}
        }))


    elif analysis_type == 'Analysis 7 - High Sales Return Ratio Customers':
    # ---------------- Analysis 7: Sales Ratio ----------------
        st.header("🔍 Analysis 7: Sales Return Ratio Deep Dive")

        # Step 1: Dynamic Filters (Tran_type, Region, Area, BRAND)
        st.subheader("Filter: Transaction Type, Region, Area, and BRAND (Dynamic)")
        filtered_df = df.copy()

        # Create 4-column layout
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tx_options = ['All', 'ER', 'DR', 'NE', 'SR']
            selected_tx = st.selectbox("Select Tran_type", tx_options)

        with col2:
            # Region options depend on current filtered_df (after tx filter)
            df_tx = filtered_df if selected_tx=='All' else filtered_df[filtered_df['Tran_type']==selected_tx]
            region_opts = ['All'] + sorted(df_tx['Region'].dropna().unique().tolist())
            selected_region = st.selectbox("Select Region", region_opts)

        with col3:
            # Area options depend on tx + region
            df_rg = df_tx if selected_region=='All' else df_tx[df_tx['Region']==selected_region]
            area_opts = ['All'] + sorted(df_rg['Area'].dropna().unique().tolist())
            selected_area = st.selectbox("Select Area", area_opts)

        with col4:
            # BRAND options depend on tx + region + area
            df_ar = df_rg if selected_area=='All' else df_rg[df_rg['Area']==selected_area]
            brand_opts = ['All'] + sorted(df_ar['BRAND'].dropna().unique().tolist())
            selected_brand = st.selectbox("Select BRAND", brand_opts)

        # Apply all filters in one go
        df_filt = filtered_df.copy()
        if selected_tx    != 'All': df_filt = df_filt[df_filt['Tran_type'] == selected_tx]
        if selected_region!= 'All': df_filt = df_filt[df_filt['Region']    == selected_region]
        if selected_area  != 'All': df_filt = df_filt[df_filt['Area']      == selected_area]
        if selected_brand != 'All': df_filt = df_filt[df_filt['BRAND']     == selected_brand]

        # Step 2: Build returns_df and sales_df
        returns_df = df_filt[df_filt['Tran_type'].isin(['ER','SR','DR','NE'])].copy()
        sales_df   = df_filt[df_filt['Tran_type'].isin(['INV','IC'])].copy()

        # Step 3: Summarize per customer
        return_summary = returns_df.groupby('CUST_NAME')['Return_qty'] .sum().reset_index(name='Total_Returns')
        sales_summary  = sales_df  .groupby('CUST_NAME')['sale_qty']  .sum().reset_index(name='Total_Sales')

        # Merge & compute ratio
        ratio_df = pd.merge(return_summary, sales_summary, on='CUST_NAME', how='inner')
        ratio_df['Sales_Ratio'] = ratio_df['Total_Returns'] / ratio_df['Total_Sales']
        ratio_df = ratio_df[ratio_df['Total_Sales']>0]

        # Step 4: Sort & pick top 10
        ratio_df = ratio_df.sort_values('Sales_Ratio', ascending=False)
        top_n = 10
        top_ratio_df = ratio_df.head(top_n)

        # Step 5: Add metadata for context
        meta = df_filt[['CUST_NAME','Region','Area','BRAND']].drop_duplicates('CUST_NAME')
        top_ratio_df = top_ratio_df.merge(meta, on='CUST_NAME', how='left')

        # Step 6: Display table
        st.subheader(f"Top {top_n} Customers by Sales Return Ratio")
        st.dataframe(
            top_ratio_df[[
                'CUST_NAME','Region','Area','BRAND',
                'Total_Returns','Total_Sales','Sales_Ratio'
            ]]
            .rename(columns={'CUST_NAME':'Customer'})
            .style.format({
                'Total_Returns':'{:,.0f}',
                'Total_Sales':'{:,.0f}',
                'Sales_Ratio':'{:.2%}'
            })
        )

        # Step 7: Plot bar chart
        st.subheader(f"📊 Top {top_n} Customers with Highest Return Ratio")
        fig = px.bar(
            top_ratio_df,
            x='CUST_NAME', y='Sales_Ratio',
            hover_data=['Total_Returns','Total_Sales'],
            color='CUST_NAME',
            text=top_ratio_df['Sales_Ratio'].map(lambda x: f"{x:.1%}"),
            title="Sales Return Ratio by Customer"
        )
        fig.update_traces(textposition='outside', showlegend=False)
        fig.update_layout(
            yaxis_title="Sales Return Ratio",
            xaxis_title="Customer",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Step 8: Download
        csv = top_ratio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Sales Ratio Data as CSV",
            data=csv,
            file_name="analysis7_sales_ratio.csv",
            mime="text/csv"
        )
