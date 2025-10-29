from  userResources import generate_user_recommendations
from getTrainingSet.utils.trainingSet_utils import clear_cache
if __name__ == "__main__":
    generate_user_recommendations()

    clear_cache()
    # 清理缓存
