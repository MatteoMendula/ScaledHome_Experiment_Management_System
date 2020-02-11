import sys
import inspect

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

sys.path.append('../')
from dataset_utils import create_lists_pairs

def train_model(model, x_train, y_train):
    print(model, x_train, y_train)
    model.fit(x_train, y_train)


def split_dataset_train_val_test(dataset_uri, feature_cols, target_cols, prediction_index):
    x, y = create_lists_pairs(dataset_uri, feature_cols, target_cols, prediction_index, as_numpy=True)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.125,
        shuffle=False
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def scale(train, val, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    return train, val, test


def evaluate(model, dataset_parts, dataset_part):
    if dataset_part == 'train':
        x, y = dataset_parts['train']['x'], dataset_parts['train']['y']
    elif dataset_part == 'val':
        x, y = dataset_parts['val']['x'], dataset_parts['val']['y']
    elif dataset_part == 'test':
        x, y = dataset_parts['test']['x'], dataset_parts['test']['y']
    else:
        raise Exception('The dataset part should be train, val or test.')
    y_hat = model.predict(x)
    return mean_squared_error(y, y_hat, multioutput='raw_values')


def get_hyperparameters(class_instace):
    attributes = inspect.getmembers(class_instace, lambda a:not(inspect.isroutine(a)))
    return list(a for a in attributes if not(a[0].startswith('__') 
            and a[0].endswith('__')) 
            and not(a[0].startswith('data_set_parts'))
            and not(a[0].startswith('x'))
            and not(a[0].startswith('y'))
            )