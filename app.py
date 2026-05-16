import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# 2. تحميل الموديل وتخزينه في الكاش
@st.cache_resource
def load_model():
    return joblib.load('churn_model (1).joblib')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"خطأ في تحميل ملف الموديل: {e}")
    model_loaded = False

# 3. العنوان الرئيسي للـ Dashboard
st.title("📱 AI-Enhanced Data Pipeline for Customer Churn Prediction")
st.markdown("---")

if model_loaded:
    tab1, tab2 = st.tabs(["🔮 Single Customer Prediction", "📈 Financial Feasibility & ROI"])
    
    with tab1:
        st.header("Single Customer Risk Assessment")
        st.subheader("Enter Customer Usage & Network Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Financial & Recharge")
            montant = st.number_input("Montant (Recharge Amount)", min_value=0.0, value=1000.0)
            frequence_rech = st.number_input("Frequence Rech", min_value=0.0, value=5.0)
            revenue = st.number_input("Revenue", min_value=0.0, value=1500.0)
            arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=500.0)
            frequence = st.number_input("Frequence (Activity)", min_value=0.0, value=10.0)
            
        with col2:
            st.markdown("### 📊 Usage & Behavior")
            data_volume = st.number_input("Data Volume (MB)", min_value=0.0, value=250.0)
            on_net = st.number_input("On Net (Calls within network)", min_value=0.0, value=50.0)
            orange = st.number_input("Orange (Calls to Orange)", min_value=0.0, value=15.0)
            tigo = st.number_input("Tigo (Calls to Tigo)", min_value=0.0, value=5.0)
            regularity = st.number_input("Regularity (Days active / 90)", min_value=0, max_value=90, value=30)
            freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=2.0)

        with col3:
            st.markdown("### 📡 Network Quality & Region Data (2019)")
            region_tower_count = st.number_input("Region Tower Count", min_value=0, value=12)
            region_avg_range = st.number_input("Region Avg Range", min_value=0.0, value=2.5)
            region_avg_samples = st.number_input("Region Avg Samples", min_value=0.0, value=150.0)
            region_coverage_index = st.number_input("Region Coverage Index", min_value=0.0, max_value=1.0, value=0.85)
            region_network_quality_score = st.number_input("Region Network Quality Score", min_value=0.0, max_value=100.0, value=75.0)
            arr_network_quality_score = st.number_input("Arr Network Quality Score", min_value=0.0, max_value=100.0, value=70.0)
            
        st.markdown("---")
        
        if st.button("Analyze Customer Churn Risk", type="primary"):
            # تجميع البيانات في مصفوفة Numpy معزولة تماماً عن الأسماء والـ Headers
            input_features = np.array([[
                montant, 
                frequence_rech, 
                revenue, 
                arpu_segment, 
                frequence, 
                data_volume, 
                on_net, 
                orange, 
                tigo, 
                regularity, 
                freq_top_pack, 
                region_tower_count, 
                region_avg_range, 
                region_avg_samples, 
                region_coverage_index, 
                region_network_quality_score, 
                arr_network_quality_score
            ]], dtype=np.float64)
            
            # تنفيذ التنبؤ باستخدام المصفوفة المجردة
            prediction = model.predict(input_features)[0]
            probability = model.predict_proba(input_features)[0][1]
            
            st.subheader("Analysis Results")
            if prediction == 1 or probability > 0.5:
                st.error(f"⚠️ High Risk Client! Churn Probability: {probability:.2%}")
                st.progress(float(probability))
                st.write("**Recommendation:** Trigger retention campaign immediately (e.g., offer a customized top pack or discount).")
            else:
                st.success(f"✅ Loyal Client. Churn Probability: {probability:.2%}")
                st.progress(float(probability))
                st.write("**Recommendation:** Maintain standard relationship and monitor usage regularity.")

    with tab2:
        st.header("Financial Feasibility & ROI Analysis")
        st.write("Simulation of saving customers using this predictive AI model vs standard operations.")
        
        c1, c2 = st.columns(2)
        with c1:
            avg_customer_value = st.number_input("Average Monthly Revenue per User (ARPU)", value=150.0)
            total_churn_customers = st.number_input("Simulated Churn Customers Count", value=1000)
        with c2:
            retention_cost = st.number_input("Cost of Retention Offer per Customer", value=30.0)
            model_accuracy_retention = st.slider("Estimated Model Success Rate in Retention (%)", 0, 100, 70)
            
        total_lost_revenue = total_churn_customers * avg_customer_value
        saved_customers = int(total_churn_customers * (model_accuracy_retention / 100))
        revenue_saved = saved_customers * avg_customer_value
        total_campaign_cost = total_churn_customers * retention_cost
        net_profit_roi = revenue_saved - total_campaign_cost
        roi_percentage = (net_profit_roi / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
        
        st.markdown("### 📊 Simulated Business Impact")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Potential Revenue Lost", f"${total_lost_revenue:,.2f}")
        kpi2.metric("Revenue Saved via AI", f"${revenue_saved:,.2f}", f"+{saved_customers} Clients")
        kpi3.metric("Net ROI Benefit", f"${net_profit_roi:,.2f}", f"{roi_percentage:.1f}% ROI")
