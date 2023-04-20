from logging_utils import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_feature_importance(model, feature: pd.DataFrame,
                                 num: int=20, save: bool=0):
    logger = setup_logger()
    logger.debug("visualize_feature_importance function is executing...")
    
    feature_importance = pd.DataFrame({"Value": model.feature_importances_,
                                       "Feature": feature.columns})
    
    total_value = feature_importance["Value"].sum()
    feature_importance["Value"] = (feature_importance["Value"] / total_value) * 100
    
    logger.info(f"Feature importances for the machine learning model:\n{feature_importance}")
    
    plt.figure(figsize=(15, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", 
                data=feature_importance.sort_values(by="Value", 
                                                    ascending=False)[0: num])
    plt.title("Features")
    plt.tight_layout()
    
    if save: plt.savefig("feature_importances.png")
    
    plt.show(block=True)
