import streamlit as st
import pandas as pd
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
    analysis_type = st.sidebar.selectbox("Choose", ['Analysis 1 - Overall View', 'Analysis 2 - Deep Dive into each Return Type', 'Analysis 3 - BRAND Wise Returns', 'Analysis 5 - Customers with only ER and no SR and NE', 'Analysis 6 - Closing Stock Trend', 'Analysis 7 - High Sales Return Ratio Customers'])

    return_types = ['ER', 'NE', 'DR', 'SR']
    return_df = df[df['Tran_type'].isin(return_types)]
    return_counts = return_df['Tran_type'].value_counts().reindex(return_types).fillna(0)

    if analysis_type == 'Analysis 1 - Overall View':
        st.subheader("1. Overall Return Type Distribution")

        # Bar Chart (only raw values)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(return_counts.index, return_counts.values, color='skyblue')
        ax.set_title('Total No. of Return Transactions by Type (ER, NE, DR, SR)')
        ax.set_xlabel('Return Type')
        ax.set_ylabel('Count')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', ha='center', va='bottom')

        st.pyplot(fig)
        download_chart(fig, "Overall.png")

        # Summary Table: % in Top Return Types vs Overall
        st.write("### Summary Table: % Contribution in Top 4 vs All Return Types")

        all_return_counts = df['Tran_type'].value_counts()
        top_types = return_counts.index.tolist()

        summary_table = pd.DataFrame({
            'Return Type': top_types,
            '% in Top 4 Types': [(count / return_counts.sum()) * 100 for count in return_counts],
            '% in All Types': [(all_return_counts[t] / all_return_counts.sum()) * 100 if t in all_return_counts else 0 for t in top_types]
        })

        summary_table = summary_table.set_index('Return Type')
        st.dataframe(summary_table.style.format({
            '% in Top 4 Types': "{:.1f}%",
            '% in All Types': "{:.1f}%"
        }))

        # Basic stats as earlier
        st.write("### Total Quantities by Return Type")
        for t in return_types:
            total_sales = df[df['Tran_type'] == t]['sale_qty'].sum()
            total_returns = df[df['Tran_type'] == t]['Return_qty'].sum()
            total_gross = df[df['Tran_type'] == t]['gross_amount'].sum()

            st.markdown(f"""
            **{t}**
            - Total Sale Quantity: {total_sales:,.0f}
            - Total Return Quantity: {total_returns:,.0f}
            - Total Gross Amount: ₹{total_gross:,.0f}
            """)

        st.divider()

    elif analysis_type == 'Analysis 2 - Deep Dive into each Return Type':
        # Deep Dive Based on Selected Return Type
        selected_type = st.selectbox("Select Return Type to Deep Dive", return_types)

        df_type = df[df['Tran_type'] == selected_type]

        # 2. REGION LEVEL
        st.subheader(f"2. Top 10 Regions for Return Type: {selected_type}")
        top_regions = df_type['Region'].value_counts().nlargest(10)
        all_regions = df_type['Region'].value_counts()

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_regions.index, y=top_regions.values, ax=ax_r, palette='Reds')
        ax_r.set_ylabel("Return Count")
        ax_r.set_title(f"Top 10 Regions by {selected_type} Returns")

        for i, v in enumerate(top_regions.values):
            ax_r.text(i, v + 50, str(int(v)), ha='center', fontweight='bold')

        st.pyplot(fig_r)
        download_chart(fig_r, "region_return_chart.png")

        # Summary table for regions
        region_summary = pd.DataFrame({
            '% Contribution in Top 10': (top_regions / top_regions.sum() * 100).round(2),
            #'% Contribution in All Regions': (top_regions / all_regions.sum() * 100).round(2)
        })
        st.write("### Region Contribution Summary")
        st.dataframe(region_summary)

        # 3. AREA LEVEL
        selected_region = st.selectbox("Select Region to See Area Breakdown", top_regions.index)
        df_region = df_type[df_type['Region'] == selected_region]
        top_areas = df_region['Area'].value_counts().nlargest(10)
        all_areas = df_region['Area'].value_counts()

        st.subheader(f"3. Top 10 Areas in {selected_region} for {selected_type} Returns")
        fig_a, ax_a = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_areas.index, y=top_areas.values, ax=ax_a, color='orange')
        ax_a.set_ylabel("Return Count")
        ax_a.set_title(f"Top 10 Areas in {selected_region} by {selected_type} Returns")
        plt.xticks(rotation=45)

        for i, v in enumerate(top_areas.values):
            ax_a.text(i, v + 10, str(int(v)), ha='center')

        st.pyplot(fig_a)
        download_chart(fig_a, "area_return_chart.png")

        # Summary table for areas
        area_summary = pd.DataFrame({
            '% Contribution in Top 10': (top_areas / top_areas.sum() * 100).round(2),
            #'% Contribution in All Areas in Region': (top_areas / all_areas.sum() * 100).round(2)
        })
        st.write("### Area Contribution Summary")
        st.dataframe(area_summary)

        # 4. CUSTOMER LEVEL
        selected_area = st.selectbox("Select Area to See CUST_NAME Breakdown", top_areas.index)
        df_area = df_region[df_region['Area'] == selected_area]
        customer_group = df_area.groupby('CUST_NAME')[['Return_qty', 'gross_amount']].sum()
        top_customers = customer_group.sort_values(by='Return_qty', ascending=False).head(10)
        all_customers = customer_group.sort_values(by='Return_qty', ascending=False)

        st.subheader(f"4. Top 10 Customers in {selected_area} by {selected_type} Return Quantity")
        fig_cust, ax_cust = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_customers['Return_qty'], y=top_customers.index, ax=ax_cust, palette='Blues_d')
        ax_cust.set_xlabel("Return Quantity")
        ax_cust.set_title(f"Top 10 Customers in {selected_area} by {selected_type} Return Quantity")
        st.pyplot(fig_cust)
        download_chart(fig_cust, "customer_return_chart.png")

        # Summary table for customers
        customer_summary = pd.DataFrame({
            '% Contribution in Top 10': (top_customers['Return_qty'] / top_customers['Return_qty'].sum() * 100).round(2),
            '% Contribution in All Customers in Area': (top_customers['Return_qty'] / all_customers['Return_qty'].sum() * 100).round(2)
        })
        st.write("### CUST_NAME Contribution Summary")
        st.dataframe(customer_summary)

        # Existing return/gross display
        st.write("### Return and Gross Amount for Top Customers")
        st.dataframe(top_customers.style.format({'Return_qty': '{:,.0f}', 'gross_amount': '₹{:,.0f}'}))

        # Download last chart
        buf = BytesIO()
        fig_cust.savefig(buf, format="png")
        st.download_button("📥 Download CUST_NAME Graph as PNG", data=buf.getvalue(), file_name="top_customers.png", mime="image/png")

    elif analysis_type == 'Analysis 3 - BRAND Wise Returns':
        st.subheader("🔍 Analysis 3: BRAND-based Deep Dive (Top 5 Only)")

        return_types = ['ER', 'NE', 'DR', 'SR']
        df_returns = df[df['Tran_type'].isin(return_types)]

        # Step 1: Top 5 Brands by Return Quantity
        top_brands = (
            df_returns.groupby('BRAND')['Return_qty']
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index
        )
        df_top5 = df_returns[df_returns['BRAND'].isin(top_brands)]

        pivot = df_top5.pivot_table(
            index='BRAND', columns='Tran_type', values='Return_qty',
            aggfunc='sum', fill_value=0
        )
        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)
        pivot_chart = pivot.drop(columns='Total')

        st.markdown("#### 📦 Top 5 Brands by Return Quantity (ER, DR, SR, NE)")
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_chart[return_types].plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')

        for i, brand in enumerate(pivot_chart.index):
            y_offset = 0
            for rt in return_types:
                height = pivot_chart.loc[brand, rt]
                if height > 0:
                    ax.text(i, y_offset + height/2, f"{int(height)}", ha='center', va='center', fontsize=9)
                    y_offset += height
            ax.text(i, y_offset + 10, f"Total: {int(y_offset)}", ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_title("Top 5 Brands Return Qty by Return Type", fontsize=14)
        ax.set_ylabel("Return Quantity")
        ax.set_xlabel("BRAND")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title="Return Type")
        st.pyplot(fig)
        download_chart(fig, "Brand_Return.png")

        # Summary table
        full_brand_returns = df_returns.groupby('BRAND')['Return_qty'].sum()
        summary = pd.DataFrame({
            'Return Qty': pivot['Total'],
            '% of Top 5': (pivot['Total'] / pivot['Total'].sum() * 100).round(2),
            '% of All Brands': (pivot['Total'] / full_brand_returns.sum() * 100).round(2)
        })
        st.dataframe(summary.style.format({'Return Qty': '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Brands': '{:.2f}%'}))

        # Step 2: Region-wise for selected brand
        selected_brand = st.selectbox("Select a BRAND", top_brands)
        brand_df = df_returns[df_returns['BRAND'] == selected_brand]

        region_data = (
            brand_df.groupby(['Region', 'Tran_type'])['Return_qty']
            .sum()
            .unstack(fill_value=0)
        )
        region_data['Total'] = region_data.sum(axis=1)
        region_data = region_data.sort_values('Total', ascending=False).drop(columns='Total')
        region_top = region_data.head(5)

        st.markdown(f"#### 🌎 Region-wise Return for {selected_brand}")
        fig, ax = plt.subplots(figsize=(12, 6))
        region_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (idx, row) in enumerate(region_top.iterrows()):
            cumulative = 0
            for rt in return_types:
                height = row[rt]
                if height > 0:
                    ax.text(i, cumulative + height/2, f"{int(height)}", ha='center', va='center', fontsize=8)
                    cumulative += height
            ax.text(i, cumulative + 10, f"{int(cumulative)}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_title(f'{selected_brand} Return Qty by Region', fontsize=14)
        ax.set_ylabel('Return Quantity')
        ax.set_xlabel('Region')
        ax.legend(title='Return Type')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        download_chart(fig, "Region_Return.png")

        region_summary = pd.DataFrame({
            'Return Qty': region_top.sum(axis=1),
            '% of Top 5': (region_top.sum(axis=1) / region_top.sum().sum() * 100).round(2),
            '% of All Regions': (region_top.sum(axis=1) / brand_df.groupby('Region')['Return_qty'].sum().sum() * 100).round(2)
        })
        st.dataframe(region_summary.style.format({'Return Qty': '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Regions': '{:.2f}%'}))

        selected_region = st.selectbox("Select a Region", region_top.index)

        # Step 3: Area-wise
        area_df = brand_df[brand_df['Region'] == selected_region]
        area_data = (
            area_df.groupby(['Area', 'Tran_type'])['Return_qty']
            .sum()
            .unstack(fill_value=0)
        )
        area_data['Total'] = area_data.sum(axis=1)
        area_data = area_data.sort_values('Total', ascending=False).drop(columns='Total')
        area_top = area_data.head(5)

        st.markdown(f"#### 🏙️ Area-wise Return for {selected_brand} in {selected_region}")
        fig, ax = plt.subplots(figsize=(12, 6))
        area_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (idx, row) in enumerate(area_top.iterrows()):
            cumulative = 0
            for rt in return_types:
                height = row[rt]
                if height > 0:
                    ax.text(i, cumulative + height/2, f"{int(height)}", ha='center', va='center', fontsize=8)
                    cumulative += height
            ax.text(i, cumulative + 10, f"{int(cumulative)}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_title(f'{selected_brand} Return Qty by Area in {selected_region}', fontsize=14)
        ax.set_ylabel('Return Quantity')
        ax.set_xlabel('Area')
        ax.legend(title='Return Type')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        download_chart(fig, "Area_Return.png")

        area_summary = pd.DataFrame({
            'Return Qty': area_top.sum(axis=1),
            '% of Top 5': (area_top.sum(axis=1) / area_top.sum().sum() * 100).round(2),
            '% of All Areas': (area_top.sum(axis=1) / area_df.groupby('Area')['Return_qty'].sum().sum() * 100).round(2)
        })
        st.dataframe(area_summary.style.format({'Return Qty': '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Areas': '{:.2f}%'}))

        selected_area = st.selectbox("Select an Area", area_top.index)

        # Step 4: CUST_NAME-wise
        cust_df = brand_df[(brand_df['Region'] == selected_region) & (brand_df['Area'] == selected_area)]
        cust_data = (
            cust_df.groupby(['CUST_NAME', 'Tran_type'])['Return_qty']
            .sum()
            .unstack(fill_value=0)
        )
        cust_data['Total'] = cust_data.sum(axis=1)
        cust_data = cust_data.sort_values('Total', ascending=False).drop(columns='Total')
        cust_top = cust_data.head(5)

        st.markdown(f"#### 🧍 CUST_NAME-wise Return in {selected_area}, {selected_region}")
        fig, ax = plt.subplots(figsize=(12, 6))
        cust_top.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        for i, (idx, row) in enumerate(cust_top.iterrows()):
            cumulative = 0
            for rt in return_types:
                height = row[rt] if rt in row else 0
                if height > 0:
                    ax.text(i, cumulative + height/2, f"{int(height)}", ha='center', va='center', fontsize=7)
                    cumulative += height
            ax.text(i, cumulative + 5, f"{int(cumulative)}", ha='center', va='bottom', fontweight='bold', fontsize=8)
        ax.set_title(f'{selected_brand} Return by CUST_NAME in {selected_area}', fontsize=14)
        ax.set_ylabel('Return Quantity')
        ax.set_xlabel('CUST_NAME')
        ax.legend(title='Return Type')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        download_chart(fig, "Customer_Return.png")

        cust_summary = pd.DataFrame({
            'Return Qty': cust_top.sum(axis=1),
            '% of Top 5': (cust_top.sum(axis=1) / cust_top.sum().sum() * 100).round(2),
            '% of All Customers': (cust_top.sum(axis=1) / cust_df.groupby('CUST_NAME')['Return_qty'].sum().sum() * 100).round(2)
        })
        st.dataframe(cust_summary.style.format({'Return Qty': '{:,.0f}', '% of Top 5': '{:.2f}%', '% of All Customers': '{:.2f}%'}))

        selected_customer = st.selectbox("Select a CUST_NAME", cust_top.index)

        # Step 5: Summary
        cust_summary_df = df[
            (df['CUST_NAME'] == selected_customer) &
            (df['BRAND'] == selected_brand) &
            (df['Region'] == selected_region) &
            (df['Area'] == selected_area) &
            (df['Tran_type'].isin(return_types))
        ]
        total_sales = cust_summary_df['sale_qty'].sum()
        total_returns = cust_summary_df['Return_qty'].sum()
        total_gross = cust_summary_df['gross_amount'].sum()

        st.markdown(f"### 📊 Summary for {selected_customer}")
        st.markdown(f"- **Total Sales Quantity:** {int(total_sales)}")
        st.markdown(f"- **Total Return Quantity:** {int(total_returns)}")
        st.markdown(f"- **Total Gross Amount:** ₹{int(total_gross)}")

    elif analysis_type == 'Analysis 5 - Customers with only ER and no SR and NE':
    # Step 1: Filter only for return types
        return_types = ['ER', 'NE', 'DR', 'SR']
        df_returns = df[df['Tran_type'].isin(return_types)]

        # Step 2: Group by customer and get unique return types
        customer_return_types = df_returns.groupby('CUST_NAME')['Tran_type'].unique()

        # Step 3: Identify customers who only returned ER
        customers_only_er = customer_return_types[customer_return_types.apply(lambda x: set(x) == {'ER'})].index.tolist()

        # Step 4: Filter the dataframe for these customers (optional — to inspect details)
        df_customers_only_er = df_returns[df_returns['CUST_NAME'].isin(customers_only_er)]

        # Show or display in Streamlit
        st.subheader("Customers Who Only Returned ER (Expired Returns)")
        st.write(f"Total customers: {len(customers_only_er)}")
        st.dataframe(df_customers_only_er[['CUST_NAME', 'Region', 'Area', 'BRAND', 'Return_qty', 'gross_amount']].drop_duplicates())

    elif analysis_type == "Analysis 6 - Closing Stock Trend":
        st.subheader("Analysis 6: SR/NE Returns vs Stock Buildup (Trailing 3-Month ClsStk Check)")

        # Upload Sheet 2
        uploaded_file_sheet2 = st.file_uploader("Upload Excel File - Sheet 2 (Closing Stock Data)", type=["xlsx"], key="sheet2")

        if uploaded_file_sheet2:
            # Read both sheets
            sheet1 = df  # Original sheet
            sheet2 = load_data(uploaded_file_sheet2)  # Sheet 2 for ClsStk data

            st.header("Analysis 6: Returns vs 3-Month Trailing Stock Buildup")

            # Parse StartDt column in sheet2
            sheet2["StartDt"] = pd.to_datetime(sheet2["StartDt"], format="%d-%m-%Y", errors="coerce")
            sheet2["Month_Year"] = sheet2["StartDt"].dt.to_period("M")
            sheet2["Month_Name"] = sheet2["StartDt"].dt.strftime('%B %Y')

            # Get all available valid months from Sheet 2 (after April)
            available_months = sorted(sheet2["Month_Year"].unique())
            valid_months = [m for m in available_months if m >= pd.Period("2024-07")]

            # Create a dropdown with full month names
            month_mapping = {pd.Period(m, freq='M'): pd.Period(m, freq='M').strftime('%B %Y') for m in valid_months}
            reverse_month_mapping = {v: k for k, v in month_mapping.items()}
            selected_month_str = st.selectbox("Select Month (for Analysis 6)", list(month_mapping.values()))
            selected_month = reverse_month_mapping[selected_month_str]

            # Filter returns for selected month and types SR/NE
            sheet1["INV_DATE"] = pd.to_datetime(sheet1["INV_DATE"])
            sheet1["Month_Year"] = sheet1["INV_DATE"].dt.to_period("M")
            returns_filtered = sheet1[
                (sheet1["Tran_type"].isin(["SR", "NE"])) &
                (sheet1["Month_Year"] == selected_month)
            ]

            # Get top 10 customers by SR+NE quantity
            top_customers = (
                returns_filtered.groupby(["CUST_NAME", "Tran_type"])["Return_qty"]
                .sum().unstack(fill_value=0)
            )
            top_customers["Total_Returns"] = top_customers.sum(axis=1)
            top_customers = top_customers.sort_values("Total_Returns", ascending=False).head(10)

            # Bar chart (stacked SR/NE)
            # Enhanced stacked bar chart with value labels
            top_customers_plot = top_customers.reset_index()

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=top_customers_plot["CUST_NAME"],
                y=top_customers_plot["SR"],
                name="SR",
                marker_color='indianred',
                text=top_customers_plot["SR"],
                textposition='inside'
            ))

            fig_bar.add_trace(go.Bar(
                x=top_customers_plot["CUST_NAME"],
                y=top_customers_plot["NE"],
                name="NE",
                marker_color='steelblue',
                text=top_customers_plot["NE"],
                textposition='inside'
            ))

            # Add total return labels on top of the bars
            for i, cust in enumerate(top_customers_plot["CUST_NAME"]):
                total = top_customers_plot.loc[i, "Total_Returns"]
                fig_bar.add_annotation(
                    x=cust,
                    y=total + 1,  # offset for clarity
                    text=str(total),
                    showarrow=False,
                    font=dict(color="white", size=12)
                )

            fig_bar.update_layout(
                barmode='stack',
                title=f"Top 10 Customers by SR + NE Returns in {selected_month.strftime('%B %Y')}",
                xaxis_title="CUST_NAME",
                yaxis_title="Return Quantity",
                legend_title="Return Type",
                height=500
            )

            st.plotly_chart(fig_bar, use_container_width=True)


            # Closing stock for previous 3 months
            prev_months = [
                selected_month - 1,
                selected_month - 2,
                selected_month - 3
            ]

            closing_stock_df = sheet2[
                sheet2["Month_Year"].isin(prev_months)
            ]

            # Prepare ClsStk table per customer
            table_data = {"CUST_NAME": []}
            for m in prev_months:
                col_name = m.strftime('%b-%Y')  # e.g., Jul-2024
                table_data[col_name] = []

            for cust in top_customers.index:
                table_data["CUST_NAME"].append(cust)
                cust_data = closing_stock_df[closing_stock_df["CUST_NAME"] == cust]
                for m in prev_months:
                    val = cust_data[cust_data["Month_Year"] == m]["ClsStk"].sum()
                    table_data[m.strftime('%b-%Y')].append(val)

            table_df = pd.DataFrame(table_data)
            st.markdown("### CUST_NAME-wise Closing Stock for Previous 3 Months")
            st.dataframe(table_df, use_container_width=True)

    elif analysis_type == 'Analysis 7 - High Sales Return Ratio Customers':
        # ---------------- Analysis 7: Sales Ratio ----------------

        st.header("🔍 Analysis 7: Sales Return Ratio Deep Dive")

        # Filter 1: Dynamic Region, Area, and BRAND Filters
        st.subheader("Filter: Region, Area, and BRAND (Dynamic)")
        filtered_df = df.copy()

        # Create 3-column layout
        col1, col2, col3 = st.columns(3)

        # Region dropdown
        with col1:
            region_options = sorted(filtered_df['Region'].dropna().unique())
            selected_region = st.selectbox("Select Region", ['All'] + region_options)

        # Area dropdown
        with col2:
            if selected_region != 'All':
                area_options = sorted(filtered_df[filtered_df['Region'] == selected_region]['Area'].dropna().unique())
            else:
                area_options = sorted(filtered_df['Area'].dropna().unique())
            selected_area = st.selectbox("Select Area", ['All'] + area_options)

        # BRAND dropdown
        with col3:
            temp_df = filtered_df.copy()
            if selected_region != 'All':
                temp_df = temp_df[temp_df['Region'] == selected_region]
            if selected_area != 'All':
                temp_df = temp_df[temp_df['Area'] == selected_area]
            brand_options = sorted(temp_df['BRAND'].dropna().unique())
            selected_brand = st.selectbox("Select BRAND", ['All'] + brand_options)

        # Apply all selected filters
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        if selected_area != 'All':
            filtered_df = filtered_df[filtered_df['Area'] == selected_area]
        if selected_brand != 'All':
            filtered_df = filtered_df[filtered_df['BRAND'] == selected_brand]

        # Filter for Return types and Sale types
        returns_df = filtered_df[filtered_df['Tran_type'].isin(['ER', 'SR', 'DR', 'NE'])].copy()
        sales_df = filtered_df[filtered_df['Tran_type'].isin(['INV', 'IC'])].copy()

        # Aggregate by CUST_NAME
        return_summary = returns_df.groupby('CUST_NAME')['Return_qty'].sum().reset_index(name='Total_Returns')
        sales_summary = sales_df.groupby('CUST_NAME')['sale_qty'].sum().reset_index(name='Total_Sales')

        # Merge and calculate sales ratio
        ratio_df = pd.merge(return_summary, sales_summary, on='CUST_NAME', how='inner')
        ratio_df['Sales_Ratio'] = ratio_df['Total_Returns'] / ratio_df['Total_Sales']
        ratio_df = ratio_df[ratio_df['Total_Sales'] > 0]

        # Sort by sales ratio descending
        ratio_df = ratio_df.sort_values(by='Sales_Ratio', ascending=False)

        # Top 10 customers
        top_n = 10
        top_ratio_df = ratio_df.head(top_n)

        # Merge metadata for context (Region, Area, BRAND)
        metadata_cols = ['CUST_NAME', 'Region', 'Area', 'BRAND']
        metadata = filtered_df[metadata_cols].drop_duplicates(subset='CUST_NAME')
        top_ratio_df = pd.merge(top_ratio_df, metadata, on='CUST_NAME', how='left')

        # Final display table
        st.subheader(f"Top {top_n} Customers by Sales Return Ratio")
        st.dataframe(top_ratio_df[['CUST_NAME', 'Region', 'Area', 'BRAND', 'Total_Returns', 'Total_Sales', 'Sales_Ratio']])

        # Plot
        st.subheader(f"📊 Top {top_n} Customers with Highest Return Ratio")
        fig = px.bar(
            top_ratio_df,
            x='CUST_NAME',
            y='Sales_Ratio',
            hover_data=['Total_Returns', 'Total_Sales'],
            color='CUST_NAME',
            text=top_ratio_df['Sales_Ratio'].apply(lambda x: f"{x:.1%}"),
            title="Sales Return Ratio by CUST_NAME",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Sales Return Ratio", xaxis_title="CUST_NAME", height=500)

        st.plotly_chart(fig, use_container_width=True)

        # Add download option
        csv_download = top_ratio_df.to_csv(index=False)
        st.download_button("Download Sales Ratio Data as CSV", data=csv_download, file_name="sales_ratio_analysis.csv", mime="text/csv")
