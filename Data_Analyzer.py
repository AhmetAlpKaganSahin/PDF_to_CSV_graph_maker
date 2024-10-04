import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tabula
import io


# Function to extract tables from a specified page in PDF
def pdf_to_tables(pdf_file, page_number):
    # Convert PDF to DataFrames using tabula, extracting multiple tables
    dfs = tabula.read_pdf(pdf_file, pages=str(page_number), encoding="ISO-8859-9", multiple_tables=True)
    if not dfs:
        st.error("No tables found on the selected page.")
        return None

    return dfs  # Return a list of DataFrames


# Function to preprocess data
def preprocess_data(df, value_col_idx):
    df.columns = df.columns.str.strip()

    st.write("### Raw Data Before Preprocessing")
    st.dataframe(df)

    for col in df.columns[value_col_idx:]:
        df[col] = df[col].apply(lambda x: str(x).replace('.', '').replace(',', '.') if pd.notnull(x) else x)

    for col in df.columns[value_col_idx:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write("### Data After Preprocessing")
    st.dataframe(df)

    return df


# Streamlit app
st.title('PDF to CSV Converter and Data Analyzer')

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
page_number = st.number_input("Enter the page number to extract table from", min_value=1, value=1)

if uploaded_file:
    # Extract multiple tables from the specified page
    dfs = pdf_to_tables(uploaded_file, page_number)

    if dfs is not None and len(dfs) > 0:
        st.write(f"### {len(dfs)} tables found on the selected page")

        # Let user select which table to plot
        table_options = [f"Table {i + 1}" for i in range(len(dfs))]
        selected_table = st.selectbox("Select the table to plot", options=table_options)

        # Get the selected table index
        selected_table_idx = table_options.index(selected_table)

        # Display the selected table
        df = dfs[selected_table_idx]
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Graph type selection
        plot_type = st.selectbox(
            "Select plot type",
            ['Bar Graph', 'Line Graph', 'Scatter Plot', 'Pie Chart', 'Double Bar Graph']
        )

        # Input column index based on the selected plot type
        if plot_type == 'Pie Chart':
            value_col_input = st.number_input("Enter column index for Pie Chart values", min_value=0, value=1)
        elif plot_type == 'Double Bar Graph':
            label_col_input = st.number_input("Enter column index for X-axis labels", min_value=0, value=0)
            value1_col_input = st.number_input("Enter column index for First Bar values", min_value=0, value=1)
            value2_col_input = st.number_input("Enter column index for Second Bar values", min_value=0, value=2)
        else:
            label_col_input = st.number_input("Enter column index for X-axis labels", min_value=0, value=0)
            value1_col_input = st.number_input("Enter column index for Values", min_value=0, value=1)

        # Preprocess data
        if plot_type == 'Pie Chart':
            df = preprocess_data(df, value_col_input)
        elif plot_type == 'Double Bar Graph':
            df = preprocess_data(df, value1_col_input)
        else:
            df = preprocess_data(df, value1_col_input)

        try:
            # Data selection based on plot type
            if plot_type == 'Pie Chart':
                x_data = df.iloc[:, 0].astype(str).tolist()  # Labels for pie chart from the first column
                y_data = df.iloc[:, value_col_input].tolist()

                # Remove NaN values and filter out rows with 'TOPLAM' or 'TOTAL'
                valid_indexes = ~pd.isnull(y_data) & (~df.iloc[:, 0].str.contains('TOPLAM|TOTAL', case=False, na=False))

                x_data = [x_data[i] for i in range(len(x_data)) if valid_indexes[i]]
                y_data = [y_data[i] for i in range(len(y_data)) if valid_indexes[i]]


                # Define autopct function to show percentage and amounts for values greater than 10%
                def custom_autopct(pct, all_vals):
                    absolute = int(pct / 100. * sum(all_vals))  # Calculate the actual value
                    if pct > 10:
                        return f'{pct:.1f}%\n({absolute})'  # Display percentage and value in double brackets
                    else:
                        return f'{pct:.1f}%'


                if st.button("Create Pie Chart"):
                    if len(x_data) == 0 or len(y_data) == 0:
                        st.error("No valid data to display.")
                    else:
                        fig, ax = plt.subplots()
                        wedges, texts, autotexts = ax.pie(y_data, labels=x_data,
                                                          autopct=lambda pct: custom_autopct(pct, y_data))

                        # Format the font size for the percentage and value display
                        for text in autotexts:
                            text.set_fontsize(10)

                        st.pyplot(fig)

                        # Save the plot as a PNG file
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        st.info(
                            "You can download your pie chart and the csv to ask gpt for explanation or analization of the data.")
                        st.download_button(
                            label="Download Pie Chart as PNG",
                            data=buf,
                            file_name="pie_chart.png",
                            mime="image/png"
                        )


            elif plot_type == 'Double Bar Graph':

                x_data = df.iloc[:, label_col_input].astype(str).tolist()
                y_data_1 = df.iloc[:, value1_col_input].tolist()
                y_data_2 = df.iloc[:, value2_col_input].tolist()

                # Remove NaN values
                valid_indexes_1 = ~pd.isnull(y_data_1)
                valid_indexes_2 = ~pd.isnull(y_data_2)
                valid_indexes = valid_indexes_1 & valid_indexes_2  # Keep only rows where both bars have valid data

                x_data = [x_data[i] for i in range(len(x_data)) if valid_indexes[i]]
                y_data_1 = [y_data_1[i] for i in range(len(y_data_1)) if valid_indexes[i]]
                y_data_2 = [y_data_2[i] for i in range(len(y_data_2)) if valid_indexes[i]]

                if st.button("Create Double Bar Graph"):

                    st.write(f"### {plot_type}")
                    fig, ax = plt.subplots()
                    bar_width = 0.35
                    index = range(len(x_data))

                    bars1 = ax.bar(index, y_data_1, bar_width, label='Bar 1')
                    bars2 = ax.bar([i + bar_width for i in index], y_data_2, bar_width, label='Bar 2')

                    ax.set_xlabel('Category')
                    ax.set_ylabel('Values')
                    ax.set_xticks([i + bar_width / 2 for i in index])
                    ax.set_xticklabels(x_data, rotation=45, ha='right')  # Rotate x-axis labels to vertical

                    ax.legend()

                    # Annotate each bar with its value
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            yval = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom',
                                    rotation=90)

                    st.pyplot(fig)

                    # Save the plot as a PNG file
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    st.info(
                        "You can download your double bar graph and the csv to ask gpt for explanation or analization of the data.")

                    st.download_button(
                        label="Download Double Bar Graph as PNG",
                        data=buf,
                        file_name="double_bar_graph.png",
                        mime="image/png"
                    )


            else:

                x_data = df.iloc[:, label_col_input].astype(str).tolist()
                y_data = df.iloc[:, value1_col_input].tolist()
                # Remove NaN values for all other plots
                valid_indexes = ~pd.isnull(y_data)
                x_data = [x_data[i] for i in range(len(x_data)) if valid_indexes[i]]
                y_data = [y_data[i] for i in range(len(y_data)) if valid_indexes[i]]

                if len(x_data) == 0 or len(y_data) == 0:

                    st.error("Insufficient data to plot the graph.")

                else:

                    if st.button(f"Create {plot_type}"):

                        st.write(f"### {plot_type}")
                        fig, ax = plt.subplots()

                        # Plot according to selected type

                        if plot_type == 'Bar Graph':

                            bars = ax.bar(x_data, y_data, color='blue')
                            ax.set_xlabel('Category')
                            ax.set_ylabel('Values')

                            # Annotate each bar with its value
                            for bar in bars:
                                yval = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2, yval,round(yval, 2), ha='center', va='bottom')

                        elif plot_type == 'Line Graph':
                            ax.plot(x_data, y_data, marker='o', color='blue')
                            ax.set_xlabel('Category')
                            ax.set_ylabel('Values')
                            ax.grid(True)

                        elif plot_type == 'Scatter Plot':
                            ax.scatter(x_data, y_data, color='blue')
                            ax.set_xlabel('Category')
                            ax.set_ylabel('Values')
                            ax.grid(True)

                        # Show the plot
                        st.pyplot(fig)

                        # Save the plot as a PNG file
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        st.info("You can download your plot and the csv to ask gpt for explanation or analization of the data.")

                        st.download_button(
                            label=f"Download {plot_type} as PNG",
                            data=buf,
                            file_name=f"{plot_type.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )

        except Exception as e:
            st.error(f"An error occurred: {e}")
        # Download the selected table as a CSV

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=';')  # Change the separator to ';'
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name='processed_data.csv',
            mime='text/csv'
        )
