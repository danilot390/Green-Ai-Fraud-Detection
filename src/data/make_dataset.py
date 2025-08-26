import os 
import torch
import joblib 

from src.data.preprocess import get_preprocessed_data, PROCESSED_DATA_PATH
from src.utils.config_parser import load_config

if __name__ == '__main__':
    print("Running make dataset...")
    
    print('\n --- Processing Credit Card Fraud Dataset ---')
    try:
        ccf_processed_dir = PROCESSED_DATA_PATH['credit_card_fraud_path']

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = \
            get_preprocessed_data(dataset_name="credit_card_fraud", target_column="Class")
        
        # Save Tensors 
        torch.save(X_train, os.path.join(ccf_processed_dir, 'X_train.pt'))
        torch.save(y_train, os.path.join(ccf_processed_dir, 'y_train.pt'))
        torch.save(X_val, os.path.join(ccf_processed_dir, 'X_val.pt'))
        torch.save(y_val, os.path.join(ccf_processed_dir, 'y_val.pt'))
        torch.save(X_test, os.path.join(ccf_processed_dir, 'X_test.pt'))
        torch.save(y_test, os.path.join(ccf_processed_dir, 'y_test.pt'))

        # Save Scaler
        joblib.dump(scaler, os.path.join(ccf_processed_dir,'scaler.joblib'))

        # Save Feature Names:
        with open(os.path.join(ccf_processed_dir,'feature_names.txt'), 'w')as f:
            for item in feature_names:
                f.write('%s\n'% item)

        print(f'Credit Card Fraud processed data and loaders save to: ')
    except Exception as e:
        print(f'Error processing Credit Card Fraud data: {e}')

    print('\n --- Processing Synthetic Dataset ---')
    try:
        sd_processed_dir = PROCESSED_DATA_PATH['synthetic_data_path']

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = \
            get_preprocessed_data(dataset_name="synthetic_data", target_column="isFraud")
        
        # Save Tensors 
        torch.save(X_train, os.path.join(sd_processed_dir, 'X_train.pt'))
        torch.save(y_train, os.path.join(sd_processed_dir, 'y_train.pt'))
        torch.save(X_val, os.path.join(sd_processed_dir, 'X_val.pt'))
        torch.save(y_val, os.path.join(sd_processed_dir, 'y_val.pt'))
        torch.save(X_test, os.path.join(sd_processed_dir, 'X_test.pt'))
        torch.save(y_test, os.path.join(sd_processed_dir, 'y_test.pt'))

        # Save Scaler
        joblib.dump(scaler, os.path.join(sd_processed_dir,'scaler.joblib'))

        # Save Feature Names:
        with open(os.path.join(sd_processed_dir,'feature_names.txt'), 'w')as f:
            for item in feature_names:
                f.write('%s\n'% item)

        print(f'Synthetic Credit Card Fraud processed data and loaders save to: {sd_processed_dir}')
    except Exception as e:
        print(f'Error processing synthetic Credit Card Fraud data: {e}')
