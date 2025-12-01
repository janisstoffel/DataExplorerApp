
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

def get_base_figure(array_config, shift_vector=(0, 0, 0)):
    sx, sy, sz = shift_vector
    
    # Get nominal markers from config
    markers = array_config.get_nominal_markers()
    
    # Coordinate Transformation
    # Input System (Nominal): X_in, Y_in, Z_in
    # Target System (Plotly): X_plot, Y_plot, Z_plot
    
    def transform(p):
        # Apply shift first
        x = p['x'] - sx
        y = p['y'] - sy
        z = p['z'] - sz
        
        if array_config.axis_mapping == 'phantom':
            # Phantom Mapping:
            # X_plot = X_in
            # Y_plot = Z_in
            # Z_plot = -Y_in
            return {
                'x': x,
                'y': z,
                'z': -y,
                'label': p.get('label', ''),
                'color': p.get('color', 'blue')
            }
        else:
            # Instrument Array Mapping:
            # "nominal - y must be x, nomial Z must be z again"
            # X_plot = -Y_in
            # Y_plot = X_in (Inferred to complete rotation)
            # Z_plot = Z_in
            return {
                'x': -y,
                'y': x,
                'z': z,
                'label': p.get('label', ''),
                'color': p.get('color', 'blue')
            }

    # Apply transformation
    markers_trans = {k: transform(v) for k, v in markers.items()}
    
    # Create traces
    data = []
    
    # Add markers
    x_markers = [m['x'] for m in markers_trans.values()]
    y_markers = [m['y'] for m in markers_trans.values()]
    z_markers = [m['z'] for m in markers_trans.values()]
    text_markers = list(markers_trans.keys())
    
    data.append(go.Scatter3d(
        x=x_markers, y=y_markers, z=z_markers,
        mode='markers+text',
        marker=dict(size=10, color='lightblue'),
        text=text_markers,
        textposition="top center",
        name='Nominal Markers'
    ))
    
    # Note: CB Points and Other Points were specific to Phantom.
    # If other arrays have them, we should add them to the config.
    # For now, we only plot the main CS markers defined in the config.

    layout = go.Layout(
        title=f"{array_config.name} 3D Model",
        uirevision='constant',
        scene=dict(
            xaxis_title='X (Nominal X)' if array_config.axis_mapping == 'phantom' else 'X (Nominal -Y)',
            yaxis_title='Y (Nominal Z)' if array_config.axis_mapping == 'phantom' else 'Y (Nominal X)',
            zaxis_title='Z (Nominal -Y)' if array_config.axis_mapping == 'phantom' else 'Z (Nominal Z)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return go.Figure(data=data, layout=layout)

def generate_figure(array_config, df, params):
    # params is a dict containing:
    # expansion_factor, center_val, point_size_mult, color_scale_name, sigma_val, selected_point, selected_tags, search_term
    
    expansion_factor = params.get('expansion_factor', 1)
    center_val = params.get('center_val', 'global')
    point_size_mult = params.get('point_size_mult', 1)
    color_scale_name = params.get('color_scale_name', 'default')
    sigma_val = params.get('sigma_val', 3)
    selected_point = params.get('selected_point')
    selected_tags = params.get('selected_tags')
    search_term = params.get('search_term')
    
    status_msg = f"Loaded {len(df)} points."
    
    # Calculate shift vector based on selected center
    shift_vector = (0, 0, 0)
    nominals = array_config.get_nominal_markers()
    
    if center_val in nominals:
        nom = nominals[center_val]
        shift_vector = (nom['x'], nom['y'], nom['z'])
    
    # Get base figure
    fig = get_base_figure(array_config, shift_vector=shift_vector)
    
    if df.empty:
        return fig, "No data loaded."

    # --- Data Preparation ---
    
    # 1. Apply Search Filter
    if search_term:
        df = df[df['FileName'].str.contains(search_term, case=False, na=False)]
        status_msg = f"Found {len(df)} points matching '{search_term}'."

    if df.empty:
            return fig, "No matching data."

    # 2. Determine Masks for Highlight, Visible, Grey
    mask_highlight = pd.Series([False] * len(df), index=df.index)
    mask_visible = pd.Series([True] * len(df), index=df.index)
    mask_grey = pd.Series([False] * len(df), index=df.index)
    
    has_selection = False
    
    if selected_tags:
        has_selection = True
        folder_tags = [t.split(':', 1)[1] for t in selected_tags if t.startswith('folder:')]
        file_tags = [t.split(':', 1)[1] for t in selected_tags if t.startswith('file:')]
        
        # Default to hiding everything if tags are present
        mask_visible = pd.Series([False] * len(df), index=df.index)
        mask_grey = pd.Series([True] * len(df), index=df.index)
        
        if folder_tags:
            mask_in_folder = df['Folder'].apply(lambda x: any(str(x).startswith(f) for f in folder_tags))
            mask_visible = mask_visible | mask_in_folder
            mask_grey = mask_grey & ~mask_in_folder
            
        if file_tags:
            mask_in_file = df['FileName'].isin(file_tags)
            mask_highlight = mask_highlight | mask_in_file
            mask_visible = mask_visible | mask_in_file
            mask_grey = mask_grey & ~mask_in_file
    
    # Handle Click Selection
    if selected_point:
        has_selection = True
        ts = selected_point[2]
        mask_click = df['Timestamp'] == ts
        mask_highlight = mask_highlight | mask_click
        mask_visible = mask_visible | mask_click
        mask_grey = mask_grey & ~mask_click
        
    # If no selection at all, everything is visible (Normal View)
    if not has_selection:
        mask_visible = pd.Series([True] * len(df), index=df.index)
        mask_grey = pd.Series([False] * len(df), index=df.index)
        mask_highlight = pd.Series([False] * len(df), index=df.index)

    # --- Calculations (Outliers, Expansion) ---
    
    # Normalize timestamps for color gradient
    df['Timestamp_Int'] = df['Timestamp'].astype(np.int64)
    min_ts = df['Timestamp_Int'].min()
    max_ts = df['Timestamp_Int'].max()
    if max_ts == min_ts:
        df['Color_Scale'] = 1.0
    else:
        df['Color_Scale'] = (df['Timestamp_Int'] - min_ts) / (max_ts - min_ts)

    # Identify Outliers and Calculate Averages
    df['IsOutlier'] = False
    averages_data = []
    feature_centroids = {}
    
    # We calculate averages for ALL features present in the data, not just nominals
    # This allows CB points to have averages too
    unique_features = df['Feature'].unique()
    
    for feat in unique_features:
        mask = df['Feature'] == feat
        coords = df.loc[mask, ['X', 'Y', 'Z']].values
        centroid = np.mean(coords, axis=0)
        feature_centroids[feat] = centroid
        dists = np.linalg.norm(coords - centroid, axis=1)
        std_dev = np.std(dists)
        if std_dev > 0:
            outliers = dists > (sigma_val * std_dev)
            df.loc[mask, 'IsOutlier'] = outliers
        
        # Expanded Centroid
        # If feature has a nominal, expand from nominal. Else expand from origin (or just shift)
        if feat in nominals:
            nom = nominals[feat]
            nx, ny, nz = nom['x'], nom['y'], nom['z']
            cx, cy, cz = centroid
            ex = nx + expansion_factor * (cx - nx) - shift_vector[0]
            ey = ny + expansion_factor * (cy - ny) - shift_vector[1]
            ez = nz + expansion_factor * (cz - nz) - shift_vector[2]
        else:
            # For non-nominal points (like CB), we just shift them relative to the center
            # Or maybe we should expand them from their own centroid?
            # Let's expand from their own centroid relative to the shift vector?
            # If we don't have a nominal, we can't do "Nominal + Expansion * (Measured - Nominal)"
            # We can do "Shifted + Expansion * (Measured - Shifted)"? No.
            # Let's just shift them for now.
            cx, cy, cz = centroid
            ex = cx - shift_vector[0]
            ey = cy - shift_vector[1]
            ez = cz - shift_vector[2]
            
        # Transform average point for plotting
        if array_config.axis_mapping == 'phantom':
            tx, ty, tz = ex, ez, -ey
        else:
            tx, ty, tz = -ey, ex, ez
            
        averages_data.append({'x': tx, 'y': ty, 'z': tz, 'label': f"{feat} Avg", 'orig_x': ex, 'orig_y': ey, 'orig_z': ez})

    # Apply Expansion Factor
    def expand_row(row):
        feat = row['Feature']
        if feat in nominals:
            nom = nominals[feat]
            nx, ny, nz = nom['x'], nom['y'], nom['z']
            mx, my, mz = row['X'], row['Y'], row['Z']
            ex = nx + expansion_factor * (mx - nx) - shift_vector[0]
            ey = ny + expansion_factor * (my - ny) - shift_vector[1]
            ez = nz + expansion_factor * (mz - nz) - shift_vector[2]
        else:
            ex = row['X'] - shift_vector[0]
            ey = row['Y'] - shift_vector[1]
            ez = row['Z'] - shift_vector[2]
        
        # Apply Axis Transformation
        if array_config.axis_mapping == 'phantom':
            return pd.Series([ex, ez, -ey])
        else:
            return pd.Series([-ey, ex, ez])

    expanded_coords = df.apply(expand_row, axis=1)
    df['X_exp'] = expanded_coords[0]
    df['Y_exp'] = expanded_coords[1]
    df['Z_exp'] = expanded_coords[2]
    
    # --- Plotting ---
    
    # 1. Grey Layer
    df_grey = df[mask_grey]
    if not df_grey.empty:
        fig.add_trace(go.Scatter3d(
            x=df_grey['X_exp'],
            y=df_grey['Y_exp'],
            z=df_grey['Z_exp'],
            mode='markers',
            marker=dict(size=2 * point_size_mult, color='lightgrey', opacity=0.1),
            text=df_grey.apply(lambda row: f"Batch: {row['BatchNumber']}<br>Time: {row['Timestamp']}<br>{row['Feature']}", axis=1),
            customdata=df_grey[['BatchNumber', 'Feature', 'Timestamp', 'X', 'Y', 'Z']].values,
            name='Other Data'
        ))
        
    # 2. Visible Layer (Normal + Outliers) - Excluding Highlighted
    mask_visible_only = mask_visible & ~mask_highlight
    df_vis = df[mask_visible_only]
    
    if not df_vis.empty:
        df_outliers = df_vis[df_vis['IsOutlier']]
        df_normal = df_vis[~df_vis['IsOutlier']]
        
        # Normal Points
        marker_dict = dict(size=2 * point_size_mult)
        
        # If a folder is selected, user wants points to be blue
        is_folder_selection = False
        if selected_tags:
            folder_tags_check = [t for t in selected_tags if t.startswith('folder:')]
            if folder_tags_check:
                is_folder_selection = True
        
        if is_folder_selection:
            marker_dict['color'] = 'blue'
            marker_dict['showscale'] = False
        else:
            marker_dict['showscale'] = True
            marker_dict['colorbar'] = dict(title="Time", tickvals=[0, 1], ticktext=["Old", "New"], len=0.5)
            
            if color_scale_name == 'default':
                def get_default_color(t):
                    r = int(211 + t * (0 - 211))
                    g = int(211 + t * (0 - 211))
                    b = int(211 + t * (139 - 211))
                    return f'rgb({r},{g},{b})'
                marker_dict['color'] = df_normal['Color_Scale'].apply(get_default_color)
                marker_dict['showscale'] = False
            else:
                marker_dict['color'] = df_normal['Color_Scale']
                marker_dict['colorscale'] = color_scale_name
        
        # Differentiate CS vs CB points
        # We can use symbols or just rely on text
        # Let's use circles for CS and diamonds for CB
        df_normal_cs = df_normal[df_normal['Feature'].str.startswith('CS')]
        df_normal_cb = df_normal[df_normal['Feature'].str.startswith('CB')]
        
        if not df_normal_cs.empty:
            # Prepare marker dict for CS
            marker_dict_cs = marker_dict.copy()
            if 'color' in marker_dict_cs and isinstance(marker_dict_cs['color'], pd.Series):
                marker_dict_cs['color'] = marker_dict_cs['color'].loc[df_normal_cs.index]

            fig.add_trace(go.Scatter3d(
                x=df_normal_cs['X_exp'],
                y=df_normal_cs['Y_exp'],
                z=df_normal_cs['Z_exp'],
                mode='markers',
                marker=marker_dict_cs,
                text=df_normal_cs.apply(lambda row: f"Batch: {row['BatchNumber']}<br>Time: {row['Timestamp']}<br>{row['Feature']}", axis=1),
                customdata=df_normal_cs[['BatchNumber', 'Feature', 'Timestamp', 'X', 'Y', 'Z']].values,
                name='Measured CS'
            ))
            
        if not df_normal_cb.empty:
            # Copy marker dict and change symbol
            marker_dict_cb = marker_dict.copy()
            marker_dict_cb['symbol'] = 'diamond'
            if 'color' in marker_dict_cb and isinstance(marker_dict_cb['color'], pd.Series):
                marker_dict_cb['color'] = marker_dict_cb['color'].loc[df_normal_cb.index]
            
            fig.add_trace(go.Scatter3d(
                x=df_normal_cb['X_exp'],
                y=df_normal_cb['Y_exp'],
                z=df_normal_cb['Z_exp'],
                mode='markers',
                marker=marker_dict_cb,
                text=df_normal_cb.apply(lambda row: f"Batch: {row['BatchNumber']}<br>Time: {row['Timestamp']}<br>{row['Feature']}", axis=1),
                customdata=df_normal_cb[['BatchNumber', 'Feature', 'Timestamp', 'X', 'Y', 'Z']].values,
                name='Measured CB'
            ))
        
        # Outliers
        if not df_outliers.empty:
            fig.add_trace(go.Scatter3d(
                x=df_outliers['X_exp'],
                y=df_outliers['Y_exp'],
                z=df_outliers['Z_exp'],
                mode='markers',
                marker=dict(size=2 * point_size_mult, color='red', symbol='cross'),
                text=df_outliers.apply(lambda row: f"Batch: {row['BatchNumber']}<br>Time: {row['Timestamp']}<br>{row['Feature']}<br>OUTLIER", axis=1),
                customdata=df_outliers[['BatchNumber', 'Feature', 'Timestamp', 'X', 'Y', 'Z']].values,
                name='Outliers'
            ))

    # 3. Highlight Layer (Selected Kits)
    df_high = df[mask_highlight]
    if not df_high.empty:
        # Group by Timestamp (Kit) to draw lines/mesh per kit
        for ts, df_kit in df_high.groupby('Timestamp'):
            
            # Sort based on connection order
            # Filter only CS points for the loop
            df_kit_cs = df_kit[df_kit['Feature'].str.startswith('CS')]
            
            # Create a mapping for sorting
            # connection_order is like [1, 2, 4, 3]
            # We want to map any feature string containing "CS01" or "CS1" to index 0, etc.
            
            def get_sort_index(feature_name):
                # Extract number from CSxx
                match = re.search(r'CS(\d+)', feature_name)
                if match:
                    num = int(match.group(1))
                    if num in array_config.connection_order:
                        return array_config.connection_order.index(num)
                return 999 # Put at end if not found
            
            df_kit_cs['SortKey'] = df_kit_cs['Feature'].apply(get_sort_index)
            df_kit_cs = df_kit_cs.sort_values('SortKey')
            
            # Kit Points Trace (Blue)
            x_kit = df_kit_cs['X_exp'].tolist()
            y_kit = df_kit_cs['Y_exp'].tolist()
            z_kit = df_kit_cs['Z_exp'].tolist()
            
            # Close loop
            if len(x_kit) > 0:
                x_kit.append(x_kit[0])
                y_kit.append(y_kit[0])
                z_kit.append(z_kit[0])
            
            fig.add_trace(go.Scatter3d(
                x=x_kit, y=y_kit, z=z_kit,
                mode='markers+lines',
                marker=dict(size=2.5 * point_size_mult, color='blue'),
                line=dict(color='blue', width=4),
                text=df_kit_cs.apply(lambda row: f"Batch: {row['BatchNumber']}<br>Time: {row['Timestamp']}<br>{row['Feature']}", axis=1).tolist() + [""],
                customdata=df_kit_cs[['BatchNumber', 'Feature', 'Timestamp', 'X', 'Y', 'Z']].values,
                name=f'Selected Kit {ts}'
            ))
            
            # Vectors (for all points in kit, including CB)
            for _, row in df_kit.iterrows():
                feat = row['Feature']
                px, py, pz = row['X_exp'], row['Y_exp'], row['Z_exp']
                avg_point = next((p for p in averages_data if p['label'] == f"{feat} Avg"), None)
                
                if avg_point and feat in feature_centroids:
                    ax, ay, az = avg_point['x'], avg_point['y'], avg_point['z']
                    mx, my, mz = row['X'], row['Y'], row['Z']
                    cx, cy, cz = feature_centroids[feat]
                    dist = np.linalg.norm(np.array([mx, my, mz]) - np.array([cx, cy, cz]))
                    
                    fig.add_trace(go.Scatter3d(
                        x=[px, ax], y=[py, ay], z=[pz, az],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'{feat} Deviation',
                        hoverinfo='text',
                        text=f"Dev: {dist:.3f} mm"
                    ))
                    
                    mid_x, mid_y, mid_z = (px + ax) / 2, (py + ay) / 2, (pz + az) / 2
                    fig.add_trace(go.Scatter3d(
                        x=[mid_x], y=[mid_y], z=[mid_z],
                        mode='text',
                        text=[f"{dist:.2f}mm"],
                        textposition="top center",
                        textfont=dict(color='red', size=10),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Mesh (Only for CS points loop)
            if len(df_kit_cs) >= 3:
                x_mesh = df_kit_cs['X_exp'].values
                y_mesh = df_kit_cs['Y_exp'].values
                z_mesh = df_kit_cs['Z_exp'].values
                # Simple triangulation for 4 points
                i, j, k = ([0, 0], [1, 2], [2, 3]) if len(df_kit_cs) >= 4 else ([0], [1], [2])
                fig.add_trace(go.Mesh3d(
                    x=x_mesh, y=y_mesh, z=z_mesh,
                    i=i, j=j, k=k,
                    color='lightblue', opacity=0.70,
                    name='Kit Area'
                ))

    # 4. Averages Trace
    if averages_data:
            fig.add_trace(go.Scatter3d(
            x=[p['x'] for p in averages_data],
            y=[p['y'] for p in averages_data],
            z=[p['z'] for p in averages_data],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Averages',
            text=[p['label'] for p in averages_data],
            hoverinfo='text'
        ))
        
    # 5. Enforce Symmetric Axis Ranges
    all_x, all_y, all_z = [], [], []
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            all_x.extend(trace.x)
            all_y.extend(trace.y)
            all_z.extend(trace.z)
            
    if not df.empty:
        all_x.extend(df['X_exp'])
        all_y.extend(df['Y_exp'])
        all_z.extend(df['Z_exp'])
        
    if averages_data:
        all_x.extend([p['x'] for p in averages_data])
        all_y.extend([p['y'] for p in averages_data])
        all_z.extend([p['z'] for p in averages_data])
        
    if all_x:
        ax = np.array([v for v in all_x if v is not None])
        ay = np.array([v for v in all_y if v is not None])
        az = np.array([v for v in all_z if v is not None])
        max_extent = max(
            np.max(np.abs(ax)) if len(ax) > 0 else 0,
            np.max(np.abs(ay)) if len(ay) > 0 else 0,
            np.max(np.abs(az)) if len(az) > 0 else 0
        ) * 1.1
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-max_extent, max_extent]),
                yaxis=dict(range=[-max_extent, max_extent]),
                zaxis=dict(range=[-max_extent, max_extent]),
                aspectmode='cube'
            )
        )
        
    return fig, status_msg
