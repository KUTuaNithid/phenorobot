#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/bow_database.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace data {

bow_database::bow_database(bow_vocabulary* bow_vocab)
    : bow_vocab_(bow_vocab) {
    spdlog::debug("CONSTRUCT: data::bow_database");
}

bow_database::~bow_database() {
    clear();
    spdlog::debug("DESTRUCT: data::bow_database");
}

void bow_database::add_keyframe(keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);

    // Append keyframe to the corresponding index in keyframes_in_node_ list
    for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
        keyfrms_in_node_[node_id_and_weight.first].push_back(keyfrm);
    }
}

void bow_database::erase_keyframe(keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);

    // Delete keyframe from the coresponding index in keyframes_in_node_ list
    for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
        // first: node ID, second: weight
        if (!static_cast<bool>(keyfrms_in_node_.count(node_id_and_weight.first))) {
            continue;
        }
        // Obtain keyframe which shares word
        auto& keyfrms_in_node = keyfrms_in_node_.at(node_id_and_weight.first);

        // std::list::erase only accepts iterator
        for (auto itr = keyfrms_in_node.begin(), lend = keyfrms_in_node.end(); itr != lend; itr++) {
            if (*keyfrm == *(*itr)) {
                keyfrms_in_node.erase(itr);
                break;
            }
        }
    }
}

void bow_database::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    spdlog::info("clear BoW database");
    keyfrms_in_node_.clear();
}

std::vector<keyframe*> bow_database::accquire_candidates_byObject_frm(frame* qry_keyfrm) {
    std::unordered_set<keyframe*> object_candidates;
    bool found = false;
    for (const auto& keyfrm_in_node : keyfrms_in_node_) {
        const auto& keyfrms = keyfrm_in_node.second;
        for (auto& keyfrm : keyfrms) {
            for (unsigned int idx_qry = 0; idx_qry < qry_keyfrm->num_lbpos_; ++idx_qry) {
                for (unsigned int idx_krm = 0; idx_krm < keyfrm->num_lbpos_; ++idx_krm) {
                    if (qry_keyfrm->labels_.at(idx_qry) == keyfrm->labels_.at(idx_krm)) {
                        // auto x1 = keyfrm->labels_pos.at(idx_krm)(0);
                        // auto y1 = keyfrm->labels_pos.at(idx_krm)(1);
                        // auto z1 = keyfrm->labels_pos.at(idx_krm)(2);
                        // auto x2 = qry_keyfrm->labels_pos.at(idx_qry)(0);
                        // auto y2 = qry_keyfrm->labels_pos.at(idx_qry)(1);
                        // auto z2 = qry_keyfrm->labels_pos.at(idx_qry)(2);
                        // long double distance = std::sqrt(std::pow(x1 - x2, 2.0) + std::pow(y1 - y2, 2.0) + std::pow(z1 - z2, 2.0));
                        // spdlog::debug("accquire_candidates_byObject: Found same object {},{} distance is {}", qry_keyfrm->labels_.at(idx_qry), keyfrm->labels_.at(idx_krm), distance);
                        // if (distance < 0.1) {
                        //     spdlog::debug("accquire_candidates_byObject: Matched object inserthis frame");
                        //     break;
                        // }
                        object_candidates.insert(keyfrm);
                        found = true;
                    }
                }
                if (found == true) {
                    found = false;
                    break;
                }
            }
        }
    }
    spdlog::debug("acquire_loop_candidates: Found object_candidates.size() {}", object_candidates.size());
    if (object_candidates.size() > 0) {
        return std::vector<keyframe*>(object_candidates.begin(), object_candidates.end());;
    }
    return std::vector<keyframe*>();
}

std::vector<keyframe*> bow_database::accquire_candidates_byObject(keyframe* qry_keyfrm) {
    std::unordered_set<keyframe*> object_candidates;
    bool found = false;
    for (const auto& keyfrm_in_node : keyfrms_in_node_) {
        const auto& keyfrms = keyfrm_in_node.second;
        for (auto& keyfrm : keyfrms) {
            for (unsigned int idx_qry = 0; idx_qry < qry_keyfrm->num_lbpos_; ++idx_qry) {
                for (unsigned int idx_krm = 0; idx_krm < keyfrm->num_lbpos_; ++idx_krm) {
                    if (qry_keyfrm->labels_.at(idx_qry) == keyfrm->labels_.at(idx_krm)) {
                        // auto x1 = keyfrm->labels_pos.at(idx_krm)(0);
                        // auto y1 = keyfrm->labels_pos.at(idx_krm)(1);
                        // auto z1 = keyfrm->labels_pos.at(idx_krm)(2);
                        // auto x2 = qry_keyfrm->labels_pos.at(idx_qry)(0);
                        // auto y2 = qry_keyfrm->labels_pos.at(idx_qry)(1);
                        // auto z2 = qry_keyfrm->labels_pos.at(idx_qry)(2);
                        // long double distance = std::sqrt(std::pow(x1 - x2, 2.0) + std::pow(y1 - y2, 2.0) + std::pow(z1 - z2, 2.0));
                        // spdlog::debug("accquire_candidates_byObject: Found same object {},{} distance is {}", qry_keyfrm->labels_.at(idx_qry), keyfrm->labels_.at(idx_krm), distance);
                        // if (distance < 0.1) {
                        //     spdlog::debug("accquire_candidates_byObject: Matched object insert this frame id {} idx_qry {} idx_krm {}", keyfrm->id_, idx_qry, idx_krm);
                        //     object_candidates.insert(keyfrm);
                        //     found = true;
                        //     break;
                        // }
                        object_candidates.insert(keyfrm);
                        found = true;
                        break;
                    }
                }
                if (found == true) {
                    found = false;
                    break;
                }
            }
        }
    }
    spdlog::debug("acquire_loop_candidates: Found object_candidates.size() {}", object_candidates.size());
    if (object_candidates.size() > 0) {
        return std::vector<keyframe*>(object_candidates.begin(), object_candidates.end());;
    }
    return std::vector<keyframe*>();
}

std::vector<keyframe*> bow_database::acquire_loop_candidates(keyframe* qry_keyfrm, const float min_score) {
    std::lock_guard<std::mutex> lock(tmp_mtx_);

    initialize();

    // spdlog::debug("acquire_loop_candidates: qry_keyfrm->labels_.size() {}", qry_keyfrm->labels_.size());
    // if (qry_keyfrm->labels_.size() > 0) {
    //     auto candidate = accquire_candidates_byObject(qry_keyfrm);
    //     if (!candidate.empty()){
    //         spdlog::debug("acquire_loop_candidates: Using object candidate");
    //         return candidate;
    //     }
    // }

    // Step 1.
    // Count up the number of nodes, words which are shared with query_keyframe, for all the keyframes in DoW database

    // Not searching near frames of query_keyframe
    auto keyfrms_to_reject = qry_keyfrm->graph_node_->get_connected_keyframes();
    keyfrms_to_reject.insert(qry_keyfrm);

    // If there are no candidates, done
    if (!set_candidates_sharing_words(qry_keyfrm, keyfrms_to_reject)) {
        return std::vector<keyframe*>();
    }

    // Set min_num_common_words as 80 percentile of max_num_common_words
    // for the following selection of candidate keyframes.
    // (Delete frames from candidates if it has less shared words than 80% of the max_num_common_words)
    unsigned int max_num_common_words = 0;
    for (const auto& candidate : init_candidates_) {
        if (max_num_common_words < num_common_words_.at(candidate)) {
            max_num_common_words = num_common_words_.at(candidate);
        }
    }
    const auto min_num_common_words = static_cast<unsigned int>(0.8f * max_num_common_words);

    // Step 2.
    // Collect keyframe candidates which have more shared words than min_num_common_words
    // by calculating similarity score between each candidate and the query keyframe.

    // If there are no candidates, done
    if (!compute_scores(qry_keyfrm, min_num_common_words)) {
        return std::vector<keyframe*>();
    }

    // If there are no candidates, done
    if (!align_scores_and_keyframes(min_num_common_words, min_score)) {
        return std::vector<keyframe*>();
    }

    // Step 3.
    // Calculate sum of the similarity scores for each of score_keyfrm_pairs and the near frames
    // Candidate will be the frame which has the highest similarity score among the near frames

    const auto best_total_score = align_total_scores_and_keyframes(min_num_common_words, min_score);

    // Step 4.
    // Final candidates have larger total score than 75 percentile
    const float min_total_score = 0.75f * best_total_score;
    std::unordered_set<keyframe*> final_candidates;

    for (const auto& total_score_keyfrm : total_score_keyfrm_pairs_) {
        const auto total_score = total_score_keyfrm.first;
        const auto keyfrm = total_score_keyfrm.second;

        if (min_total_score < total_score) {
            final_candidates.insert(keyfrm);
        }
    }

    return std::vector<keyframe*>(final_candidates.begin(), final_candidates.end());
}

std::vector<keyframe*> bow_database::acquire_relocalization_candidates(frame* qry_frm) {
    std::lock_guard<std::mutex> lock(tmp_mtx_);

    initialize();
    // spdlog::debug("acquire_relocalization_candidates: qry_frm->labels_.size() {}", qry_frm->labels_.size());
    // if (qry_frm->labels_.size() > 0) {
    //     auto candidate = accquire_candidates_byObject_frm(qry_frm);
    //     if (!candidate.empty()){
    //         spdlog::debug("acquire_loop_candidates: Using object candidate");
    //         return candidate;
    //     }
    // }
    // Step 1.
    // Count up the number of nodes, words which are shared with query_keyframe, for all the keyframes in DoW database
    
    auto t_acc1 = std::chrono::steady_clock::now();
    // If there are no candidates, done
    if (!set_candidates_sharing_words(qry_frm)) {
        return std::vector<keyframe*>();
    }
    auto t_acc2 = std::chrono::steady_clock::now();
    auto use_time = std::chrono::duration_cast<std::chrono::duration<double>>(t_acc2 - t_acc1).count();
    spdlog::info("set_candidates_sharing_words  time {} ms", use_time*1000);

    // Set min_num_common_words as 80 percentile of max_num_common_words
    // for the following selection of candidate keyframes.
    // (Delete frames from candidates if it has less shared words than 80% of the max_num_common_words)
    unsigned int max_num_common_words = 0;
    for (const auto& candidate : init_candidates_) {
        if (max_num_common_words < num_common_words_.at(candidate)) {
            max_num_common_words = num_common_words_.at(candidate);
        }
    }
    const auto min_num_common_words = static_cast<unsigned int>(0.8f * max_num_common_words);

    // Step 2.
    // Collect keyframe candidates which have more shared words than min_num_common_words
    // by calculating similarity score between each candidate and the query keyframe.

    // If there are no candidates, done
    if (!compute_scores(qry_frm, min_num_common_words)) {
        return std::vector<keyframe*>();
    }

    // If there are no candidates, done
    if (!align_scores_and_keyframes(min_num_common_words, 0.0)) {
        return std::vector<keyframe*>();
    }

    // Step 3.
    // Calculate sum of the similarity scores for each of score_keyfrm_pairs and the near frames
    // Candidate will be the frame which has the highest similarity score among the near frames
    const auto best_total_score = align_total_scores_and_keyframes(min_num_common_words, 0.0);

    // Step 4.
    // Final candidates have larger total score than 75 percentile
    const float min_total_score = 0.75f * best_total_score;
    std::unordered_set<keyframe*> final_candidates;

    for (const auto& total_score_keyfrm : total_score_keyfrm_pairs_) {
        const auto total_score = total_score_keyfrm.first;
        const auto keyfrm = total_score_keyfrm.second;

        if (min_total_score < total_score) {
            final_candidates.insert(keyfrm);
        }
    }
    spdlog::info("final size {}", final_candidates.size());

    return std::vector<keyframe*>(final_candidates.begin(), final_candidates.end());
}

void bow_database::initialize() {
    init_candidates_.clear();
    num_common_words_.clear();
    scores_.clear();
    score_keyfrm_pairs_.clear();
    total_score_keyfrm_pairs_.clear();
}

#include <math.h>
template<typename T>
bool bow_database::check_label(const T* const qry_shot, const keyframe* keyfrm_in_node){
    // return true;
    int qry_labels = qry_shot->labels_.size();
    int key_labels = keyfrm_in_node->labels_.size();
    if (qry_labels <= 0) {
        spdlog::debug("check_label: qrt_shot has no object");
        return true;
    }
    if (key_labels <= 0) {
        spdlog::debug("check_label: keyframe has no object");
        return false;
    }
    double thres = 0.6;
    int matched_rule = ceil(qry_labels*thres);
    int matched = 0;
    for (const auto& qry:qry_shot->labels_){
        auto it = std::find(keyfrm_in_node->labels_.begin(), keyfrm_in_node->labels_.end(), qry);
        if (it != keyfrm_in_node->labels_.end()) {
            matched++;
        }
        if (matched >= matched_rule){
            spdlog::debug("check_label: {} is selected matched_rule {}", keyfrm_in_node->id_, matched_rule);
            return true;
        }
    }
    return false;

}

template<typename T>
bool bow_database::set_candidates_sharing_words(const T* const qry_shot, const std::set<keyframe*>& keyfrms_to_reject) {
    init_candidates_.clear();
    num_common_words_.clear();
    int filout = 0;
    std::lock_guard<std::mutex> lock(mtx_);

    // Get word (node index) of the query
    const auto& bow_vec = qry_shot->bow_vec_;
    // Count the number of shared words for keyframes which share the word with the query keyframe
    for (const auto& node_id_and_weight : bow_vec) {
        // first: node ID, second: weight
        // If not in the BoW database, continue
        if (!static_cast<bool>(keyfrms_in_node_.count(node_id_and_weight.first))) {
            continue;
        }
        // Get a keyframe which shares the word (node ID) with the query
        const auto& keyfrms_in_node = keyfrms_in_node_.at(node_id_and_weight.first);
        // For each keyframe, increase shared word number one by one
        for (const auto& keyfrm_in_node : keyfrms_in_node) {
            // Initialize if not in num_common_words
            if (!static_cast<bool>(num_common_words_.count(keyfrm_in_node))) {
                num_common_words_[keyfrm_in_node] = 0;
                // If far enough from the query keyframe, store it as the initial loop candidates
                if (!static_cast<bool>(keyfrms_to_reject.count(keyfrm_in_node))) {
                    init_candidates_.insert(keyfrm_in_node);
                }
            }
            // Count up the number of words
            ++num_common_words_.at(keyfrm_in_node);
        }
    }
    // spdlog::info("set_candidates_sharing_words: Before init_candidates_.size {} Object {}" , init_candidates_.size(), qry_shot->labels_.size());
    // Query has object ?
    if (qry_shot->labels_[0] == 1234) {
        std::vector<std::pair<keyframe*, int>> scores;
        int min_score = 10000;
        for (auto candidate : init_candidates_) {
            int score = 0;
            for (int i = 1;qry_shot->labels_.size();i++){
                score = score + std::abs(qry_shot->labels_[i] - candidate->labels_[i]);
            }
            scores.push_back(std::make_pair(candidate,score));
            if (min_score > score){
                min_score = score;
            }
        }
        float thres = (min_score+1)*110/100;
        init_candidates_.clear();
        for(auto score:scores){
            if(score.second <= thres){
                // Filter out
                init_candidates_.insert(score.first);
            }
        }

    }

    // spdlog::info("set_candidates_sharing_words: After init_candidates_.size {} Object {}" , init_candidates_.size(), qry_shot->labels_.size());

    return !init_candidates_.empty();
}

template<typename T>
bool bow_database::compute_scores(const T* const qry_shot, const unsigned int min_num_common_words_thr) {
    scores_.clear();

    for (const auto& candidate : init_candidates_) {
        if (min_num_common_words_thr < num_common_words_.at(candidate)) {
            // Calculate similarity score with query keyframe
            // for the keyframes which have more shared words than minimum common words
#ifdef USE_DBOW2
            const float score = bow_vocab_->score(qry_shot->bow_vec_, candidate->bow_vec_);
#else
            const float score = fbow::BoWVector::score(qry_shot->bow_vec_, candidate->bow_vec_);
#endif
            // Store score
            scores_[candidate] = score;
        }
    }

    return !scores_.empty();
}

bool bow_database::align_scores_and_keyframes(const unsigned int min_num_common_words_thr, const float min_score) {
    score_keyfrm_pairs_.clear();

    // If larger than the minimum score, store to score_keyfrm_pairs
    for (const auto& candidate : init_candidates_) {
        if (min_num_common_words_thr < num_common_words_.at(candidate)) {
            const float score = scores_.at(candidate);
            if (min_score <= score) {
                score_keyfrm_pairs_.emplace_back(std::make_pair(score, candidate));
            }
        }
    }

    return !score_keyfrm_pairs_.empty();
}

float bow_database::align_total_scores_and_keyframes(const unsigned int min_num_common_words_thr, const float min_score) {
    total_score_keyfrm_pairs_.clear();

    float best_total_score = min_score;

    for (const auto& score_keyframe : score_keyfrm_pairs_) {
        const auto score = score_keyframe.first;
        const auto keyfrm = score_keyframe.second;

        // Get near frames of keyframe
        const auto top_n_covisibilities = keyfrm->graph_node_->get_top_n_covisibilities(10);
        // Calculate the sum of scores for the near frames
        // Initialize with score since keyframe is not included in covisibility_keyframes
        float total_score = score;

        // Find a keyframe which has best similarity score with query keyframe from the near frames
        float best_score = score;
        auto best_keyframe = keyfrm;

        for (const auto& covisibility : top_n_covisibilities) {
            // Loop for which is included in the initial loop candidates and satisfies the minimum shared word number
            if (static_cast<bool>(init_candidates_.count(covisibility))
                && min_num_common_words_thr < num_common_words_.at(covisibility)) {
                // score has already been computed
                total_score += scores_.at(covisibility);
                if (best_score < scores_.at(covisibility)) {
                    best_score = scores_.at(covisibility);
                    best_keyframe = covisibility;
                }
            }
        }

        total_score_keyfrm_pairs_.emplace_back(std::make_pair(total_score, best_keyframe));

        if (best_total_score < total_score) {
            best_total_score = total_score;
        }
    }

    return best_total_score;
}

} // namespace data
} // namespace openvslam
