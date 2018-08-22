/**
 * @brief ranking_response definition. ranking_response is returned from choose_rank call.  It contains the chosen action and probability distribution.
 *
 * @file ranking_response.h
 * @author Rajan Chari et al
 * @date 2018-07-18
 */
#pragma once
#include <cstddef>

namespace reinforcement_learning {
  class api_status;
  class ranking_response_impl;

  /**
   * @brief Holds (action, probability) pairs.
   */
  struct action_prob {
    //! action id
    size_t action_id;
    //! probability associated with the action id
    float probability;
  };

  /**
   * @brief choose_rank() returns the action choice using ranking_response.
   * ranking_response contains all the actions and distribution from with the action was sampled.  It also contains the chosen action id and
   * the unique event_id representing the choice.  This unique event_id must be used to report back outcomes against this choice for the online trainer.
   */
  class ranking_response {
  public:
    ranking_response();
    ~ranking_response();
    /**
     * @brief Construct a new ranking response object.
     *
     * @param event_id The unique identifier for this interaction.  This event_id must be used when reporting the outcome for this action
     */
    ranking_response(char const* event_id);

    /**
     * @brief Unique event_id for this ranking request.
     * This event_id must be used when calling report_outcome so it can be joined with the chosen action
     * @return const char*
     */
    const char* get_event_id() const;

    /**
     * @brief Get the chosen action id.
     *
     * @param action_id Chosen action id
     * @param status Optional field with detailed string description if there is an error
     * @return int Error code
     */
    int get_chosen_action_id(size_t& action_id, api_status* status = nullptr) const; // id of the top action chosen by the model

    /**
     * @brief Set the chosen action id.  (This is set internally by the API)
     *
     * @param action_id Chosen action id
     * @param status Optional field with detailed string description if there is an error
     * @return int Error code
     */
    int set_chosen_action_id(size_t action_id, api_status* status = nullptr); // id of the top action chosen by the model

    /**
     * @brief Set the event_id.  (This is set internally by the API)
     * @param event_id
     */
    void set_event_id(const char* event_id);

    /**
     * @brief Add (action id, probability) pair to the response (This is set internally by the API)
     *
     * @param action_id
     * @param prob
     */
    void push_back(const size_t action_id, const float prob);

    /**
     * @brief Size of the action collection.
     *
     * @return size_t
     */
    size_t size() const;

    /**
     * @brief Set the model_id.
     * Every call to choose action is associated with a unique model used to predict.  A unique model_id
     * is associated with each unique model. (This is set internally by the API)
     * @param model_id
     */
    void set_model_id(const char* model_id);

    /**
     * @brief Get the model_id.
     * Every call to choose action is associated with a unique model used to predict.  A unique model_id
     * is associated with each unique model. (This is set internally by the API)
     * @return const char*
     */
    const char * get_model_id() const;

    /**
     * @brief Clear the response object so that it can be reused.
     * The goal is to reuse response without reallocating as much as possible.
     */
    void clear();

    /**
     * @brief Move construct a new ranking response object.
     * The underlying data is taken from the rvalue reference.
     */
    ranking_response(ranking_response&&) noexcept;

    /**
     * @brief Move assignment operator for ranking response.
     * The underlying data is taken from rvalue reference, and then it is cleared.
     * @return ranking_response&
     */
    ranking_response& operator=(ranking_response&&) noexcept;

    /**
     * @brief Copy constructor is removed since implementation will be deleted twice
     */
    ranking_response(const ranking_response&) = delete;

    /**
     * @brief assignment operator is removed since implementation will be deleted twice
     */
    ranking_response& operator =(const ranking_response&) = delete;
  private:
    ranking_response_impl* _pimpl; //!The actual implementation details are forwarded to this object (PIMPL pattern)

  public:
    /**
     * @brief Forward iterator class used to access the (action, probability) collection
     */
    class ranking_iterator {
    public:
      //! Construct an (action, probability) collection iterator using the ranking_response implementation
      ranking_iterator(ranking_response_impl*);
      //! Construct an (action, probability) collection iterator using the ranking_response implementation and size
      ranking_iterator(ranking_response_impl*, size_t);
      //! Move the iterator to the next element
      ranking_iterator& operator++();
      //! Inequality comparison for the iterator
      bool operator!=(const ranking_iterator& other) const;
      //! Dereferencing operator to get the (action, probability) pair
      action_prob operator*() const;
    private:
      ranking_response_impl* _p_resp_impl;
      size_t _idx;
    };
    //! Returns an iterator pointing to the first element of the (action, probability) collection
    ranking_iterator begin() const;
    //! Returns an iterator referring to the past-the-end element of the (action, probability) collection.
    ranking_iterator end() const;
  };
}
