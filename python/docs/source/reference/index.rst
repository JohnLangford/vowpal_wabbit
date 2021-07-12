API Reference
=============

.. toctree::
   vowpalwabbit.pyvw
   vowpalwabbit.sklearn
   vowpalwabbit.DFtoVW
   vowpalwabbit.pyvw.pylibvw


Usage
-----

You can use the python wrapper directly like this:

.. code-block:: python

    from vowpalwabbit import pyvw
    vw = pyvw.vw(quiet=True)
    ex = vw.example('1 | a b c')
    vw.learn(ex)
    vw.predict(ex)

Or you can use the included scikit-learn interface like this:

.. code-block:: python

    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from vowpalwabbit.sklearn_vw import VWClassifier
        # generate some data
    X, y = datasets.make_hastie_10_2(n_samples=10000, random_state=1)
    X = X.astype(np.float32)
        # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=256)
        # build model
    model = VWClassifier()
    model.fit(X_train, y_train)
        # predict model
    y_pred = model.predict(X_test)
        # evaluate model
    model.score(X_train, y_train)
    model.score(X_test, y_test)
