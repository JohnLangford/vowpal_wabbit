package org.vowpalwabbit.spark;

import java.io.Closeable;

/**
 * A wrapper for the native example data structure.
 *
 * @author Markus Cozowicz
 */
public class VowpalWabbitExample implements Closeable {
    /**
     * Initializes the native VowpalWabbitExampleWrapper data structure.
     *
     * @param vwNativePointer the associated VW instance.
     * @param isEmpty         true if this should be initialized as new empty
     *                        (=new-line) example.
     * @return pointer to the native VowpalWabbitExampleWrapper data structure.
     */
    private static native long initialize(long vwNativePointer, boolean isEmpty);

    /**
     * Frees the native resources.
     */
    private native long finish();

    /**
     * Clears the features and label.
     */
    public native void clear();

    /**
     * Adds the dense features values to the supplied namespace. The
     * {@code baseIndex} is expected to be pre-hashed (e.g. hash(namespace)).
     *
     * @param ns        the first character of the namespace.
     * @param baseIndex the base index for each of the {@code values}. This is also the namespace hash (e.g. hash(namespace))
     * @param values    the feature values.
     */
    public native void addToNamespaceDense(char ns, int baseIndex, double[] values);

    /**
     * Adds the sparse features values to the supplied namespace.
     *
     * @param ns      the first character of the namespace.
     * @param indices the indices of each corresponding feature value.
     * @param values  the feature values.
     * @deprecated use the version with baseIndex instead
     */
    @Deprecated
    public void addToNamespaceSparse(char ns, int[] indices, double[] values)
    {
        addToNamespaceSparse(ns, ns, indices, values);
    }

        /**
     * Adds the sparse features values to the supplied namespace.
     *
     * @param ns      the first character of the namespace.
     * @param baseIndex the namespace hash (e.g. hash(namespace))
     * @param indices the indices of each corresponding feature value.
     * @param values  the feature values.
     */
    public native void addToNamespaceSparse(char ns, int baseIndex, int[] indices, double[] values);

    /**
     * Set the simple label.
     *
     * @param weight weight of this example.
     * @param label  value of the label (e.g. -1, 1 for binary).
     */
    public native void setLabel(float weight, float label);

    /**
     * Set a contextual bandit (CB) label
     *
     * @param action      this is 1-based.
     * @param cost        cost assigned to the action.
     * @param probability probability with which the action was chosen.
     */
    public native void setContextualBanditLabel(int action, double cost, double probability);

    /**
     * Mark this as a shared CB label.
     */
    public native void setSharedLabel();

    /**
     * Initializes the label.
     */
    public native void setDefaultLabel();

    /**
     * Updates the associated VW model using this example.
     */
    public native void learn();

    /**
     * Gets the prediction from the current example. Useful after learning to get
     * the 1-step ahead prediction.
     *
     * @return the prediction.
     */
    public native Object getPrediction();

    /**
     * Gets the prediction from the current example.
     *
     * @return the prediction.
     */
    public native Object predict();

    @Override
    public native String toString();

    /**
     * Pointer to the native VowpalWabbitExampleWrapper data structure.
     */
    private long nativePointer;

    /**
     * Initializes the native VowpalWabbitExampleWrapper data structure.
     *
     * @param vwNativePointer the associated VW instance.
     * @param isEmpty         true if this should be initialized as new empty
     *                        (=new-line) example.
     */
    VowpalWabbitExample(long vwNativePointer, boolean isEmpty) {
        this.nativePointer = initialize(vwNativePointer, isEmpty);
    }

    /**
     * Set the simple label using a weight of 1.
     *
     * @param label value of the label (e.g. -1, 1 for binary).
     */
    public void setLabel(float label) {
        setLabel(1f, label);
    }

    /**
     * Frees the native resources
     */
    @Override
    final public void close() {
        if (this.nativePointer != 0) {
            finish();
            this.nativePointer = 0;
        }
    }
}