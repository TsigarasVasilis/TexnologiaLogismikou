import streamlit as st
import pandas as pd
import numpy as np
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import base64
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="scRNA-seq Analysis Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'adata' not in st.session_state:
        st.session_state.adata = None
    if 'adata_processed' not in st.session_state:
        st.session_state.adata_processed = None
    if 'pipeline_step' not in st.session_state:
        st.session_state.pipeline_step = 0

def load_data():
    """Load and validate data"""
    st.markdown('<div class="sub-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your H5AD file",
        type=['h5ad'],
        help="Upload a single-cell RNA-seq dataset in H5AD format"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner('Loading data...'):
                # Save uploaded file temporarily
                with open("temp_data.h5ad", "wb") as f:
                    f.write(uploaded_file.read())
                
                # Load with scanpy
                adata = sc.read_h5ad("temp_data.h5ad")
                st.session_state.adata = adata
                
                st.success(f"✅ Data loaded successfully! Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
                
                # Display data overview
                display_data_overview(adata)
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return False
    
    return False

def display_data_overview(adata):
    """Display overview of the loaded dataset"""
    st.markdown('<div class="sub-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Cells", f"{adata.n_obs:,}")
    
    with col2:
        st.metric("Number of Genes", f"{adata.n_vars:,}")
    
    with col3:
        if 'batch' in adata.obs.columns:
            n_batches = adata.obs['batch'].nunique()
            st.metric("Number of Batches", n_batches)
    
    # Show available annotations
    st.markdown("**Available Annotations:**")
    annotations_info = []
    for col in adata.obs.columns:
        n_values = adata.obs[col].nunique()
        annotations_info.append(f"• **{col}**: {n_values} unique values")
    
    st.markdown("\n".join(annotations_info))
    
    # Display sample distributions
    if len(adata.obs.columns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'batch' in adata.obs.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                batch_counts = adata.obs['batch'].value_counts()
                batch_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Cells per Batch')
                ax.set_ylabel('Number of Cells')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'celltype' in adata.obs.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                celltype_counts = adata.obs['celltype'].value_counts()
                celltype_counts.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title('Cells per Cell Type')
                ax.set_ylabel('Number of Cells')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

def preprocessing_pipeline():
    """Run preprocessing pipeline with user-defined parameters"""
    if st.session_state.adata is None:
        st.warning("⚠️ Please load data first!")
        return
    
    st.markdown('<div class="sub-header">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)
    
    # Preprocessing parameters
    with st.expander("⚙️ Preprocessing Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            min_genes = st.slider(
                "Minimum genes per cell",
                min_value=0, max_value=5000, value=600,
                help="Filter out cells that have fewer than this many genes"
            )
            
            min_cells = st.slider(
                "Minimum cells per gene",
                min_value=0, max_value=100, value=3,
                help="Filter out genes that are expressed in fewer than this many cells"
            )
        
        with col2:
            n_neighbors = st.slider(
                "Number of neighbors for UMAP",
                min_value=5, max_value=50, value=15,
                help="Number of neighbors to use for UMAP calculation"
            )
            
            n_pcs = st.slider(
                "Number of principal components",
                min_value=10, max_value=100, value=50,
                help="Number of principal components to compute"
            )
    
    if st.button("🚀 Run Preprocessing", type="primary"):
        run_preprocessing(min_genes, min_cells, n_neighbors, n_pcs)

def run_preprocessing(min_genes, min_cells, n_neighbors, n_pcs):
    """Execute the preprocessing pipeline"""
    adata = st.session_state.adata.copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Filter cells
        status_text.text("Filtering cells...")
        progress_bar.progress(10)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        st.info(f"After filtering cells: {adata.n_obs} cells remaining")
        
        # Step 2: Filter genes
        status_text.text("Filtering genes...")
        progress_bar.progress(20)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        st.info(f"After filtering genes: {adata.n_vars} genes remaining")
        
        # Step 3: Remove mitochondrial genes
        status_text.text("Removing mitochondrial genes...")
        progress_bar.progress(30)
        mito_genes = [gene for gene in adata.var_names if str(gene).startswith(('ERCC', 'MT-', 'mt-'))]
        if len(mito_genes) > 0:
            adata = adata[:, ~adata.var_names.isin(mito_genes)]
            st.info(f"Removed {len(mito_genes)} mitochondrial genes")
        
        # Step 4: Normalization
        status_text.text("Normalizing data...")
        progress_bar.progress(40)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        st.info("Normalization completed")
        
        # Step 5: Find highly variable genes
        status_text.text("Finding highly variable genes...")
        progress_bar.progress(50)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        n_hvg = sum(adata.var.highly_variable)
        st.info(f"Found {n_hvg} highly variable genes")
        
        # Step 6: Store raw data and filter by HVG
        status_text.text("Filtering by highly variable genes...")
        progress_bar.progress(60)
        adata.raw = adata
        if n_hvg > 0:
            adata = adata[:, adata.var.highly_variable]
        
        # Step 7: Scale data
        status_text.text("Scaling data...")
        progress_bar.progress(70)
        sc.pp.scale(adata, max_value=10)
        st.info("Data scaling completed")
        
        # Step 8: PCA
        status_text.text("Running PCA...")
        progress_bar.progress(80)
        actual_n_pcs = min(n_pcs, min(adata.n_obs, adata.n_vars) - 1)
        if actual_n_pcs < 1:
            st.error("Cannot run PCA with current data dimensions")
            return
        sc.tl.pca(adata, n_comps=actual_n_pcs)
        st.info("PCA completed")
        
        # Step 9: Compute neighbors and UMAP
        status_text.text("Computing neighbors and UMAP...")
        progress_bar.progress(90)
        actual_n_neighbors = min(n_neighbors, adata.n_obs - 1)
        if actual_n_neighbors < 1:
            st.error("Cannot compute neighbors with current data dimensions")
            return
        
        sc.pp.neighbors(adata, n_neighbors=actual_n_neighbors, n_pcs=actual_n_pcs)
        sc.tl.umap(adata)
        st.info("UMAP completed")
        
        progress_bar.progress(100)
        status_text.text("Preprocessing completed!")
        
        # Store processed data
        st.session_state.adata_processed = adata
        st.session_state.pipeline_step = 1
        
        st.success(f"✅ Preprocessing complete! Final dataset: {adata.n_obs} cells × {adata.n_vars} genes")
        
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {str(e)}")
        return

def visualization_analysis():
    """Create visualizations for the processed data"""
    if st.session_state.adata_processed is None:
        st.warning("⚠️ Please run preprocessing first!")
        return
    
    st.markdown('<div class="sub-header">📈 Data Visualization</div>', unsafe_allow_html=True)
    
    adata = st.session_state.adata_processed
    
    # Color options for UMAP
    color_options = ['batch', 'celltype', 'disease', 'donor', 'protocol']
    valid_colors = [c for c in color_options if c in adata.obs.columns]
    
    # Add any other categorical columns
    for col in adata.obs.select_dtypes(include=['category', 'object']).columns:
        if col not in valid_colors:
            valid_colors.append(col)
    
    if valid_colors:
        selected_color = st.selectbox(
            "Select coloring variable for UMAP:",
            valid_colors,
            help="Choose how to color the UMAP plot"
        )
        
        # Create UMAP plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            sc.pl.umap(
                adata, 
                color=selected_color, 
                ax=ax, 
                show=False, 
                legend_fontsize=10,
                legend_loc='on data'
            )
            plt.title(f'UMAP colored by {selected_color}', fontsize=16)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating UMAP plot: {str(e)}")
    
    else:
        st.warning("No suitable columns found for coloring the UMAP plot")
    
    # Quality control metrics
    st.markdown("### Quality Control Metrics")
    
    if hasattr(adata, 'obs'):
        qc_metrics = []
        if 'n_genes_by_counts' in adata.obs.columns:
            qc_metrics.append('n_genes_by_counts')
        if 'total_counts' in adata.obs.columns:
            qc_metrics.append('total_counts')
        if 'pct_counts_mt' in adata.obs.columns:
            qc_metrics.append('pct_counts_mt')
        
        if qc_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                metric = st.selectbox("Select QC metric:", qc_metrics)
            
            with col2:
                if st.button("Generate QC Plot"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    adata.obs[metric].hist(bins=50, ax=ax)
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {metric}')
                    st.pyplot(fig)
                    plt.close()

def data_integration():
    """Perform data integration using Harmony"""
    if st.session_state.adata_processed is None:
        st.warning("⚠️ Please run preprocessing first!")
        return
    
    st.markdown('<div class="sub-header">🔗 Data Integration</div>', unsafe_allow_html=True)
    
    adata = st.session_state.adata_processed
    
    # Check if batch column exists
    if 'batch' not in adata.obs.columns:
        st.warning("⚠️ No 'batch' column found in the data. Skipping integration.")
        return
    
    st.info("Performing Harmony integration to correct for batch effects...")
    
    # Integration parameters
    with st.expander("🔧 Integration Parameters"):
        theta = st.slider(
            "Theta (diversity penalty)",
            min_value=0.0, max_value=5.0, value=2.0,
            help="Diversity clustering penalty parameter. Larger values result in more diverse clusters."
        )
    
    if st.button("🔗 Run Harmony Integration", type="primary"):
        try:
            with st.spinner("Running Harmony integration..."):
                # Run Harmony
                sce.pp.harmony_integrate(adata, 'batch', theta=theta)
                
                # Recompute neighbors and UMAP with integrated data
                sc.pp.neighbors(adata, use_rep="X_pca_harmony")
                sc.tl.umap(adata)
                
                st.session_state.adata_processed = adata
                st.session_state.pipeline_step = 2
                
            st.success("✅ Harmony integration completed!")
            
            # Show before/after comparison
            st.markdown("### Integration Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**After Integration - Colored by Batch**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sc.pl.umap(adata, color='batch', ax=ax, show=False, legend_fontsize=8)
                plt.title('After Harmony Integration')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                if 'celltype' in adata.obs.columns:
                    st.markdown("**After Integration - Colored by Cell Type**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sc.pl.umap(adata, color='celltype', ax=ax, show=False, legend_fontsize=8)
                    plt.title('Cell Types after Integration')
                    st.pyplot(fig)
                    plt.close()
                    
        except Exception as e:
            st.error(f"❌ Error during integration: {str(e)}")

def differential_expression_analysis():
    """Perform differential expression analysis"""
    if st.session_state.adata_processed is None:
        st.warning("⚠️ Please run preprocessing first!")
        return
    
    st.markdown('<div class="sub-header">🧬 Differential Expression Analysis</div>', unsafe_allow_html=True)
    
    adata = st.session_state.adata_processed
    
    # Check available columns for comparison
    categorical_cols = [col for col in adata.obs.columns 
                       if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category']
    
    if not categorical_cols:
        st.warning("⚠️ No categorical columns found for differential expression analysis.")
        return
    
    # Parameters for DE analysis
    with st.expander("🔧 DE Analysis Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            groupby_col = st.selectbox(
                "Group by column:",
                categorical_cols,
                help="Column to use for grouping cells"
            )
        
        with col2:
            method = st.selectbox(
                "Statistical method:",
                ['wilcoxon', 't-test', 'logreg'],
                help="Method for differential expression testing"
            )
        
        # Get unique values in selected column
        unique_values = adata.obs[groupby_col].unique()
        
        if len(unique_values) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                test_group = st.selectbox(
                    "Test group:",
                    unique_values,
                    help="Group to test for differential expression"
                )
            
            with col2:
                reference_group = st.selectbox(
                    "Reference group:",
                    [val for val in unique_values if val != test_group],
                    help="Reference group for comparison"
                )
    
    if st.button("🧬 Run Differential Expression Analysis", type="primary"):
        try:
            with st.spinner("Running differential expression analysis..."):
                # Run DE analysis
                sc.tl.rank_genes_groups(
                    adata,
                    groupby=groupby_col,
                    method=method,
                    groups=[test_group],
                    reference=reference_group,
                    use_raw=False
                )
                
                # Extract results
                deg_result = adata.uns["rank_genes_groups"]
                degs_df = pd.DataFrame({
                    "genes": deg_result["names"][test_group],
                    "pvals": deg_result["pvals"][test_group],
                    "pvals_adj": deg_result["pvals_adj"][test_group],
                    "logfoldchanges": deg_result["logfoldchanges"][test_group],
                })
                
                st.session_state.pipeline_step = 3
                
            st.success("✅ Differential expression analysis completed!")
            
            # Display results
            display_de_results(degs_df, test_group, reference_group)
            
        except Exception as e:
            st.error(f"❌ Error during DE analysis: {str(e)}")

def display_de_results(degs_df, test_group, reference_group):
    """Display differential expression results"""
    st.markdown("### 📊 Results")
    
    # Add derived columns
    degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
    degs_df["diffexpressed"] = "NS"
    degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
    degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_up = (degs_df["diffexpressed"] == "UP").sum()
        st.metric("Upregulated Genes", n_up)
    
    with col2:
        n_down = (degs_df["diffexpressed"] == "DOWN").sum()
        st.metric("Downregulated Genes", n_down)
    
    with col3:
        n_total = len(degs_df)
        st.metric("Total Genes Tested", n_total)
    
    # Top genes tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Upregulated Genes**")
        top_up = degs_df[degs_df["diffexpressed"] == "UP"].nsmallest(10, "pvals")
        if not top_up.empty:
            st.dataframe(
                top_up[["genes", "logfoldchanges", "pvals", "pvals_adj"]].round(4),
                hide_index=True
            )
        else:
            st.info("No significantly upregulated genes found")
    
    with col2:
        st.markdown("**Top Downregulated Genes**")
        top_down = degs_df[degs_df["diffexpressed"] == "DOWN"].nsmallest(10, "pvals")
        if not top_down.empty:
            st.dataframe(
                top_down[["genes", "logfoldchanges", "pvals", "pvals_adj"]].round(4),
                hide_index=True
            )
        else:
            st.info("No significantly downregulated genes found")
    
    # Volcano plot
    st.markdown("### 🌋 Volcano Plot")
    
    # Create interactive volcano plot with plotly
    fig = px.scatter(
        degs_df,
        x="logfoldchanges",
        y="neg_log10_pval",
        color="diffexpressed",
        color_discrete_map={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
        hover_data=["genes"],
        title=f"Volcano Plot: {test_group} vs {reference_group}",
        labels={
            "logfoldchanges": "log2 Fold Change",
            "neg_log10_pval": "-log10 p-value"
        }
    )
    
    # Add threshold lines
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
    fig.add_vline(x=-1, line_dash="dash", line_color="gray")
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("### 💾 Download Results")
    
    csv = degs_df.to_csv(index=False)
    st.download_button(
        label="Download DE Results as CSV",
        data=csv,
        file_name=f"de_results_{test_group}_vs_{reference_group}.csv",
        mime="text/csv"
    )

def save_processed_data():
    """Save processed data"""
    if st.session_state.adata_processed is None:
        st.warning("⚠️ No processed data to save!")
        return
    
    st.markdown('<div class="sub-header">💾 Save Processed Data</div>', unsafe_allow_html=True)
    
    if st.button("💾 Generate Download Link", type="primary"):
        try:
            # Save to bytes
            adata = st.session_state.adata_processed
            
            # Create a temporary file first
            temp_filename = "temp_processed_data.h5ad"
            adata.write_h5ad(temp_filename)
            
            # Read the file as bytes
            with open(temp_filename, "rb") as f:
                data_bytes = f.read()
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            # Create download button
            st.download_button(
                label="📥 Download Processed Data (H5AD)",
                data=data_bytes,
                file_name="processed_data.h5ad",
                mime="application/octet-stream"
            )
            
            st.success("✅ Download link generated!")
            
        except Exception as e:
            st.error(f"❌ Error saving data: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧬 Single-Cell RNA-seq Analysis Pipeline</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    pages = [
        "📁 Data Upload",
        "🔧 Preprocessing",
        "📈 Visualization",
        "🔗 Data Integration",
        "🧬 Differential Expression",
        "💾 Save Results"
    ]
    
    selected_page = st.sidebar.radio("Select Analysis Step:", pages)
    
    # Progress indicator
    st.sidebar.markdown("### 📊 Progress")
    progress_steps = [
        "Data Loaded" if st.session_state.adata is not None else "❌ Data Not Loaded",
        "Preprocessing Complete" if st.session_state.pipeline_step >= 1 else "⏳ Preprocessing Pending",
        "Integration Complete" if st.session_state.pipeline_step >= 2 else "⏳ Integration Pending",
        "DE Analysis Complete" if st.session_state.pipeline_step >= 3 else "⏳ DE Analysis Pending"
    ]
    
    for step in progress_steps:
        if "❌" in step or "⏳" in step:
            st.sidebar.markdown(f"• {step}")
        else:
            st.sidebar.markdown(f"• ✅ {step}")
    
    # Page routing
    if selected_page == "📁 Data Upload":
        load_data()
    
    elif selected_page == "🔧 Preprocessing":
        preprocessing_pipeline()
    
    elif selected_page == "📈 Visualization":
        visualization_analysis()
    
    elif selected_page == "🔗 Data Integration":
        data_integration()
    
    elif selected_page == "🧬 Differential Expression":
        differential_expression_analysis()
    
    elif selected_page == "💾 Save Results":
        save_processed_data()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This application provides a comprehensive pipeline for single-cell RNA-seq analysis, "
        "including data preprocessing, integration, visualization, and differential expression analysis."
    )

if __name__ == "__main__":
    main()