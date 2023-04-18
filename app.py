import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colnames = [
    "Id",
    "time",
    "axial strain",
    "vol_strain",
    "corr_area",
    "vert_stress",
    "horiz_stress",
    "sample_press",
    "eff vert stress",
    "eff hor stress",
    "K"
]

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Sidebar Layout
st.sidebar.markdown("# Upload Data")

uploaded_file = st.sidebar.file_uploader("Choose consolidation data .xlsx file:")

if uploaded_file is not None:
    sheetnames = pd.ExcelFile(uploaded_file).sheet_names
    option = st.sidebar.selectbox("Select the sheet:", sheetnames)

dry_density = st.sidebar.number_input(
    "Dry Density (kN/m3)",
    min_value = 0.00,
    value = 6.96,
    step = 0.01
    )

specific_gravity = st.sidebar.number_input(
    "Specific Gravity",
    min_value = 0.0,
    value = 2.7,
    step = 0.1
    )

specimen_volume = st.sidebar.number_input(
    "Specimen Volume (cc)",
    min_value = 0.00,
    value = 202.29,
    step = 0.01
    )

specimen_height = st.sidebar.number_input(
    "Specimen Height (cm)",
    min_value = 0.00,
    value = 11.18,
    step = 0.01
    )

# Principal Zone
consol_data = pd.read_excel(
    uploaded_file,
    sheet_name = option, 
    skiprows = 22, 
    names = colnames
    )

add_bh_name = st.sidebar.checkbox('Add the name of .xlsx in graph')

# Filtrado de función logarítmica
consol_data["log_time"] = np.log10(consol_data["time"])
consol_data = consol_data[consol_data.log_time > -5]

# Obtención de relación de vacios
solid_weight = dry_density  * specimen_volume / (10 ** 6)  #kN
consol_data["delta_L"] = specimen_height * consol_data["axial strain"] / 100
consol_data["NL"] = specimen_height - consol_data["delta_L"]
consol_data["Volume"] = consol_data["corr_area"] * consol_data["NL"]
consol_data["Dry Density"] = solid_weight / (consol_data["Volume"] / (10 ** 6))
consol_data["e"] = specific_gravity * 9.81 / consol_data["Dry Density"] - 1

# Principal Layout
st.markdown("# Consolidation - Triaxial Test CKoU")

st.markdown("Results of Processed Consolidation:")
consol_data

col1, col2, col3 = st.columns([3, 1, 1])

slider_1 = col1.slider(
    'Select a left range of values (Id)',
    min_value = int(consol_data["Id"].iloc[0]),
    max_value = int(consol_data["Id"].iloc[-1]),
    value = (int(consol_data["Id"].iloc[0]), int(consol_data["Id"].iloc[-1])),
    step = 1)

col2.metric("Initial value - Time(s)", str(consol_data["time"][slider_1[0]])) 
col3.metric("Initial value - Time(s)", str(consol_data["time"][slider_1[1]])) 

col4, col5, col6 = st.columns([3, 1, 1])

slider_2 = col4.slider(
    'Select a right range of values',
    min_value = int(consol_data["Id"].iloc[0]),
    max_value = int(consol_data["Id"].iloc[-1]),
    value = (int(consol_data["Id"].iloc[0]), int(consol_data["Id"].iloc[-1])),
    step = 1)

col5.metric("Initial value - Time(s)", str(consol_data["time"][slider_2[0]])) 
col6.metric("Initial value - Time(s)", str(consol_data["time"][slider_2[1]])) 

# Plot function

def plot_consolidation(slider_1, slider_2):
    
    # Gráficas de Consolidación
    curve_1_x = consol_data["log_time"][slider_1[0]:slider_1[1]]
    curve_1_y = consol_data["axial strain"][slider_1[0]:slider_1[1]]
    curve_2_x = consol_data["log_time"][slider_2[0]:slider_2[1]]
    curve_2_y = consol_data["axial strain"][slider_2[0]:slider_2[1]]

    coefs_1 = np.polyfit(curve_1_x, curve_1_y, 1)
    coefs_2 = np.polyfit(curve_2_x, curve_2_y, 1)

    t_100 = - (coefs_1[1] - coefs_2[1]) / (coefs_1[0] - coefs_2[0])
    t_100_index = consol_data.iloc[(consol_data["log_time"]-t_100).abs().argsort()[:1]].index[0]

    curve_1_x_fit = consol_data["log_time"][slider_1[0]:t_100_index + 1]
    curve_1_y_fit = curve_1_x_fit * coefs_1[0] + coefs_1[1]
    curve_2_x_fit = consol_data["log_time"][t_100_index + 1:slider_2[1]]
    curve_2_y_fit = curve_2_x_fit * coefs_2[0] + coefs_2[1]

    new_row = pd.DataFrame(np.nan, index = [0],  columns = colnames)
    new_row["time"].iloc[0] = 10 ** t_100
    interpol_data = pd.concat([consol_data, new_row], ignore_index=True)
    interpol_data.set_index('time',
                    inplace = True)

    interpol_data.interpolate(method = 'index',
                        inplace = True,
                        limit_direction = 'forward')

    interpol_data.reset_index(inplace = True)
    
    fig, ax = plt.subplots(3, 1, figsize = (10, 12))

    # Deformación axial vs Tiempo

    ax[0].plot(consol_data["time"],
            consol_data["axial strain"],
            "-",
            color = "#519EB4",
            label = "Set de Datos: " + str(option))

    ax[0].plot(10 ** curve_1_x_fit,
            curve_1_y_fit,
            "--",
            color = "#28712B")

    ax[0].plot(10 ** curve_2_x_fit,
            curve_2_y_fit,
            "--",
            color = "#28712B")

    ax[0].plot([10 ** t_100],
            [coefs_1[0] * (t_100) + coefs_1[1]],
            "o",
            color = "#163824",
            label = "$t_{100}$ =" + str(round(10 ** t_100, 1)) + "s")

    ax[0].tick_params(axis="x",direction="in", length=4)
    ax[0].tick_params(axis="y",direction="in", length=4)
    ax[0].tick_params(axis="y", which='minor',direction="in", length=4)
    ax[0].tick_params(axis="x", which='minor',direction="in", length=4)
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')

    ax[0].legend(loc = "upper right")
    
    if add_bh_name:
        ax[0].title.set_text('Ensayo Triaxial - Etapa de Consolidación - ' + uploaded_file.name[:-5] + "\n")
    else:
        ax[0].title.set_text('Ensayo Triaxial - Etapa de Consolidación'+ "\n")
    ax[0].title.set_size(16)
    ax[0].invert_yaxis()
    ax[0].set_xscale('log')
    ax[0].set_xlabel("Tiempo (s) - Escala Logarítmica", fontsize = 11)
    ax[0].set_ylabel("Deformación Axial (%)", fontsize = 11)
    ax[0].grid(True, ls="-", color='0.9')

    # Relación de Vacios vs Tiempo
    ax[1].plot(consol_data["time"],
            consol_data["e"],
            "-",
            color = "#519EB4",
            label = "Set de Datos: " + str(option))

    ax[1].plot([10 ** t_100],
            [interpol_data["e"].iloc[-1]],
            "o",
            color = "#163824",
            label = "$e$ =" + str(round(interpol_data["e"].iloc[-1], 4)))

    ax[1].grid(True, ls="-", color='0.9')
    ax[1].legend(loc = "upper right")
    ax[1].set_ylabel("Relación de vacios (e)", fontsize = 11)
    ax[1].set_xlabel("Tiempo (s) - Escala Logarítmica", fontsize = 11)
    ax[1].set_xscale('log')
    ax[1].tick_params(axis="x",direction="in", length=4)
    ax[1].tick_params(axis="y",direction="in", length=4)
    ax[1].tick_params(axis="y", which='minor',direction="in", length=4)
    ax[1].tick_params(axis="x", which='minor',direction="in", length=4)
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')


    # Relación de Vacios vs Tiempo
    ax[2].plot(consol_data["time"],
            consol_data["eff vert stress"],
            "-",
            color = "#519EB4",
            label = "Set de Datos: " + str(option))

    ax[2].plot([10 ** t_100],
            [interpol_data["eff vert stress"].iloc[-1]],
            "o",
            color = "#163824",
            label = "$\sigma'_{v0}$ =" + str(round(interpol_data["eff vert stress"].iloc[-1], 1)) + "kPa")

    ax[2].grid(True, ls="-", color='0.9')
    ax[2].legend(loc = "upper right")
    ax[2].invert_yaxis()
    ax[2].set_ylabel("Esfuerzo Vertical Efectivo (kPa)", fontsize = 11)
    ax[2].set_xlabel("Tiempo (s) - Escala Logarítmica", fontsize = 11)
    ax[2].set_xscale('log')
    ax[2].tick_params(axis="x",direction="in", length=4)
    ax[2].tick_params(axis="y",direction="in", length=4)
    ax[2].tick_params(axis="y", which='minor',direction="in", length=4)
    ax[2].tick_params(axis="x", which='minor',direction="in", length=4)
    ax[2].yaxis.set_ticks_position('both')
    ax[2].xaxis.set_ticks_position('both')
    
    
    resultados = pd.DataFrame(
    {
    "t_100 (s)": [10 ** t_100],
    "e" : [interpol_data["e"].iloc[-1]],
    "esf_eff_vert (kPa)" : [interpol_data["eff vert stress"].iloc[-1]]
     }
    )
       
    return fig, resultados

consolidation_plot = plot_consolidation(slider_1, slider_2)

st.pyplot(consolidation_plot[0])

col1, col2 = st.columns([3, 1])

col1.subheader("Results")
col1.write(consolidation_plot[1])

with col2:
    save_svg = st.checkbox('Save the fig in .svg')
    if save_svg:
        consolidation_plot[0].savefig(uploaded_file.name[:-5]+ ".svg") 
        with open(uploaded_file.name[:-5]+ ".svg", "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=uploaded_file.name[:-5]+ ".svg",
                mime="image/svg"
            )
    save_data = st.checkbox('Save the data results in .csv')
    if save_data:
        csv = convert_df(consol_data)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=uploaded_file.name[:-5]+ "csv",
            mime='text/csv',
        )

st.markdown("Author: César Manuel Sánchez Oré (cesar.sanchez@wsp.com)")
st.markdown("Checked: 18/04/2023")