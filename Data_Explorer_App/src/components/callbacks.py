from dash import callback, Input, Output, State, ctx, html
import numpy as np
import os
from src.utils.plotting import generate_figure

def register_callbacks(array_config):
    prefix = array_config.id_prefix
    
    @callback(
        Output(f'{prefix}-file-folder-dropdown', 'options'),
        Input(f'{prefix}-refresh-data-btn', 'n_clicks')
    )
    def update_dropdown_options(n_clicks):
        df = array_config.get_data(refresh=True if n_clicks > 0 else False)
        if df.empty:
            return []
        
        if 'Folder' not in df.columns:
            return []
            
        raw_folders = df['Folder'].dropna().unique()
        
        all_folders = set()
        for folder in raw_folders:
            parts = folder.split(os.sep)
            for i in range(1, len(parts) + 1):
                all_folders.add(os.path.join(*parts[:i]))
                
        unique_folders = sorted(list(all_folders))
        folder_options = [{'label': f"📁 {f}", 'value': f"folder:{f}"} for f in unique_folders]
        
        if 'FileName' not in df.columns:
            return folder_options
            
        unique_files = sorted(df['FileName'].dropna().unique())
        file_options = [{'label': f"📄 {f}", 'value': f"file:{f}"} for f in unique_files]
        
        return folder_options + file_options

    @callback(
        Output(f'{prefix}-selected-point-store', 'data'),
        Input(f'{prefix}-graph', 'clickData'),
        State(f'{prefix}-selected-point-store', 'data')
    )
    def update_selection_store(clickData, current_data):
        if not clickData:
            return None
        
        point = clickData['points'][0]
        
        if 'customdata' in point:
            new_selection = point['customdata']
            # Compare lists/arrays safely
            # customdata is a list [Batch, Feature, Timestamp, X, Y, Z]
            if current_data == new_selection:
                return None
            return new_selection
        
        return None

    @callback(
        [Output(f'{prefix}-selection-info', 'children'),
         Output(f'{prefix}-deviation-list', 'children')],
        Input(f'{prefix}-selected-point-store', 'data')
    )
    def update_selection_info(selected_data):
        if not selected_data:
            return "Click on a point to see details.", "No kit selected."
        
        batch, feat, ts, x, y, z = selected_data
        
        df = array_config.get_data()
        if df.empty:
            return "No data.", ""
            
        # 1. Point Info
        mask = df['Feature'] == feat
        if not mask.any():
            point_info = "Feature not found."
        else:
            coords = df.loc[mask, ['X', 'Y', 'Z']].values
            centroid = np.mean(coords, axis=0)
            vec = centroid - np.array([x, y, z])
            dist = np.linalg.norm(vec)
            point_info = f"Selected: {feat} (Batch {batch}). Distance to Average: {dist:.4f} mm. Vector: ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"
        
        # 2. Kit Deviations List
        # Find all points in this kit (same Timestamp)
        kit_mask = df['Timestamp'] == ts
        df_kit = df[kit_mask].sort_values('Feature')
        
        deviation_items = []
        
        # Pre-calculate centroids for all features to avoid repeated filtering
        feature_centroids = {}
        unique_features = df['Feature'].unique()
        for f in unique_features:
            f_mask = df['Feature'] == f
            f_coords = df.loc[f_mask, ['X', 'Y', 'Z']].values
            feature_centroids[f] = np.mean(f_coords, axis=0)
            
        for _, row in df_kit.iterrows():
            f = row['Feature']
            if f in feature_centroids:
                c = feature_centroids[f]
                p = np.array([row['X'], row['Y'], row['Z']])
                d = np.linalg.norm(p - c)
                
                # Format: CS1 - AVE1 = XXmm
                # We assume Feature is like CS01, CB01 etc.
                # We can just use Feature name.
                deviation_items.append(html.Div(f"{f} - AVE = {d:.4f} mm"))
        
        return point_info, deviation_items

    @callback(
        [Output(f'{prefix}-graph', "figure"),
         Output(f'{prefix}-data-status', "children")],
        [Input(f'{prefix}-refresh-data-btn', "n_clicks"),
         Input(f'{prefix}-expansion-slider', "value"),
         Input(f'{prefix}-center-dropdown', "value"),
         Input(f'{prefix}-point-size-slider', "value"),
         Input(f'{prefix}-colorscale-dropdown', "value"),
         Input(f'{prefix}-sigma-dropdown', "value"),
         Input(f'{prefix}-selected-point-store', "data"),
         Input(f'{prefix}-file-folder-dropdown', "value"),
         Input(f'{prefix}-search-input', "value")]
    )
    def update_graph(n_clicks, expansion_factor, center_val, point_size_mult, color_scale_name, sigma_val, selected_point, selected_tags, search_term):
        refresh = False
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == f"{prefix}-refresh-data-btn":
                refresh = True
                
        df = array_config.get_data(refresh=refresh)
        
        params = {
            'expansion_factor': expansion_factor,
            'center_val': center_val,
            'point_size_mult': point_size_mult,
            'color_scale_name': color_scale_name,
            'sigma_val': sigma_val,
            'selected_point': selected_point,
            'selected_tags': selected_tags,
            'search_term': search_term
        }
        
        return generate_figure(array_config, df, params)
