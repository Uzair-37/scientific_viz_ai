"""
Download utility functions for the multi-modal AI web application.

This module provides functions to enable downloading of data, visualizations,
and model results from the Streamlit web application.
"""

import base64
import json
import pickle
import pandas as pd
import numpy as np
import plotly
from io import BytesIO
import streamlit as st

def download_button(object_to_download, download_filename, button_text):
    """
    Create a download button for downloading objects in various formats.
    
    Parameters:
    -----------
    object_to_download : various types
        The object to be downloaded (DataFrame, plot, dict, etc.)
    download_filename : str
        The filename to use for the download (including extension)
    button_text : str
        The text to display on the download button
    
    Returns:
    --------
    download_button : streamlit button
        A button that when clicked will download the object
    """
    # Determine file type from filename extension
    file_extension = download_filename.split(".")[-1].lower()
    
    # Process different object types
    if isinstance(object_to_download, pd.DataFrame):
        return _download_dataframe(object_to_download, download_filename, button_text, file_extension)
    elif isinstance(object_to_download, dict):
        return _download_dict(object_to_download, download_filename, button_text, file_extension)
    elif "plotly.graph_objs" in str(type(object_to_download)):
        return _download_plotly_figure(object_to_download, download_filename, button_text, file_extension)
    elif isinstance(object_to_download, (np.ndarray, list, tuple)):
        return _download_array(object_to_download, download_filename, button_text, file_extension)
    elif isinstance(object_to_download, str):
        return _download_text(object_to_download, download_filename, button_text)
    else:
        return _download_pickle(object_to_download, download_filename, button_text)

def _download_dataframe(df, download_filename, button_text, file_extension):
    """Create a download button for DataFrame objects."""
    if file_extension == "csv":
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
    elif file_extension == "xlsx":
        # Convert DataFrame to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    elif file_extension == "json":
        # Convert DataFrame to JSON
        json_str = df.to_json(orient="records")
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'data:file/json;base64,{b64}'
    else:
        # Default to CSV if extension not recognized
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def _download_dict(dict_obj, download_filename, button_text, file_extension):
    """Create a download button for dictionary objects."""
    if file_extension == "json":
        # Convert dict to JSON
        json_str = json.dumps(dict_obj, default=lambda o: '<not serializable>')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'data:file/json;base64,{b64}'
    else:
        # Default to JSON if extension not recognized
        json_str = json.dumps(dict_obj, default=lambda o: '<not serializable>')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'data:file/json;base64,{b64}'
        
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def _download_plotly_figure(fig, download_filename, button_text, file_extension):
    """Create a download button for Plotly figures."""
    if file_extension == "html":
        # Convert Plotly figure to HTML
        html_str = plotly.io.to_html(fig, include_plotlyjs="cdn")
        b64 = base64.b64encode(html_str.encode()).decode()
        href = f'data:text/html;base64,{b64}'
    elif file_extension == "json":
        # Convert Plotly figure to JSON
        json_str = json.dumps(fig.to_dict(), default=lambda o: '<not serializable>')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'data:file/json;base64,{b64}'
    elif file_extension == "png":
        # Convert Plotly figure to PNG
        img_bytes = plotly.io.to_image(fig, format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'data:image/png;base64,{b64}'
    elif file_extension == "svg":
        # Convert Plotly figure to SVG
        img_bytes = plotly.io.to_image(fig, format="svg")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'data:image/svg+xml;base64,{b64}'
    else:
        # Default to HTML if extension not recognized
        html_str = plotly.io.to_html(fig, include_plotlyjs="cdn")
        b64 = base64.b64encode(html_str.encode()).decode()
        href = f'data:text/html;base64,{b64}'
        
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def _download_array(array_obj, download_filename, button_text, file_extension):
    """Create a download button for array-like objects."""
    # Convert to numpy array if not already
    if not isinstance(array_obj, np.ndarray):
        array_obj = np.array(array_obj)
    
    if file_extension == "csv":
        # Convert array to CSV
        csv = pd.DataFrame(array_obj).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
    elif file_extension == "npy":
        # Save array as numpy .npy file
        output = BytesIO()
        np.save(output, array_obj)
        npy_data = output.getvalue()
        b64 = base64.b64encode(npy_data).decode()
        href = f'data:application/octet-stream;base64,{b64}'
    elif file_extension == "json":
        # Convert array to JSON
        json_str = json.dumps(array_obj.tolist())
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'data:file/json;base64,{b64}'
    else:
        # Default to CSV if extension not recognized
        csv = pd.DataFrame(array_obj).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def _download_text(text, download_filename, button_text):
    """Create a download button for text."""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'data:file/txt;base64,{b64}'
    
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def _download_pickle(obj, download_filename, button_text):
    """Create a download button for arbitrary objects using pickle."""
    # Pickle the object
    pickle_data = pickle.dumps(obj)
    b64 = base64.b64encode(pickle_data).decode()
    href = f'data:application/octet-stream;base64,{b64}'
    
    # Create download button
    return st.markdown(
        f'<a href="{href}" download="{download_filename}">{button_text}</a>',
        unsafe_allow_html=True
    )

def create_download_section(data_objects, title="Download Results"):
    """
    Create a download section with multiple download buttons.
    
    Parameters:
    -----------
    data_objects : dict
        Dictionary of objects to download with format {filename: object}
    title : str
        Title for the download section
    """
    st.subheader(title)
    
    # Create columns for download buttons
    cols = st.columns(min(len(data_objects), 4))
    
    # Add download buttons
    for i, (filename, obj) in enumerate(data_objects.items()):
        with cols[i % len(cols)]:
            button_text = f"Download {filename.split('.')[0]}"
            download_button(obj, filename, button_text)