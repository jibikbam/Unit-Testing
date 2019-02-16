/*******************************************************************************
 *
 * @file PoseGenerator.hpp
 *
 ******************************************************************************/
#pragma once

#include <chrono> // for chrono::system_clock
#include <augmenter.hpp>
#include <projmeta/projmetadata.hpp>

using std::string;
using std::vector;

/**
 * @brief
 * Class which generates pose(s) (aka perturbation(s)) for each frame according to various
 * rules. Note that it only outputs numbers and pass them to Augmenter so it can perform
 * actual perturbations.
 */
class PoseGenerator
{
public:
    /**
     * @brief
     * Structure which holds parameters for any random number generation, i.e. distribution,
     * hard limit, and standard deviation. Generated numbers cannot go beyond the hard limit
     * (-max, max) and the standard deviation is used for Gaussian distributions.
     */
    struct randParams
    {
        std::string distribution;
        double max;
        double stdDev;
    };

    /**
     * @brief
     * Structure which holds parameters to generate a pose, or perturbation. These variables
     * correspond with the variables in "Pose" struct defined in Augmenter.
     */
    struct perturbParams
    {
        randParams shift;
        randParams rotation;
        randParams forward;
        randParams sensor_yaw;
        randParams sensor_pitch;
        randParams sensor_roll;
        bool flip;
    };

    /**
     * @brief
     * Constructor for the PoseGenerator that takes in perturbation rules and sensor names
     * as input.
     *
     * @param[in] configRules   : a vector of label-parameters pairs from config.
     * @param[in] sensorNames   : a vector of sensor names from config.
     */
    PoseGenerator(const std::vector<std::pair<std::string, perturbParams>>& configRules,
                  std::vector<std::string> sensorNames, unsigned int seed);

    /**
     * @brief
     * Returns a shuffled vector of Poses that matches the passed vecUseCounts vector
     * according to rules specified in labelsFileName.
     *
     * @param[in] vecUseCounts  : a vector of the number of poses to generate per frame
     * @param[in] labelsFileName: the full path to a CSV file that contains (sensor and
     *                            semantic) video labels for each frame.
     */
    std::vector<Augmenter::Pose> generateShuffledPoses(
        std::vector<uint32_t> vecUseCounts,
        const std::string& labelsFileName);

    /**
     * @brief
     * Returns a vector of vectors of Poses that matches the passed vecUseCounts vector
     * according to rules specified in labelsFileName.
     *
     * @param[in] vecUseCounts  : a vector of the number of poses to generate per frame
     * @param[in] labelsFileName: the full path to a CSV file that contains (sensor and
     *                            semantic) video labels for each frame.
     */
    std::vector<std::vector<Augmenter::Pose>> generatePoses4vecFrames(
        std::vector<uint32_t> vecUseCounts,
        const std::string& labelsFileName);

    /**
     * @brief
     * Returns a vector of Pose for a given frame.
     *
     * @param[in] useCount      : the number of poses to generate per frame
     * @param[in] index         : the frame number to generate poses
     * @param[in] trace         : a projMetaTrace class which allows to use doLabelsMatch()
     *                            and getNumDatapoints().
     */
    std::vector<Augmenter::Pose> generatePoses4oneFrame(
        uint32_t useCount,
        uint32_t index,
        const projMetaData::projMetaTrace& trace);

    /**
     * @brief
     * Returns a Pose for a frame as specified in params.
     *
     * @param[in] params        : a structure which holds parameters to generate a pose.
     */
    Augmenter::Pose generateOnePose(const perturbParams& params);

private:
    /* Perturbation Rule which is a vector of map-perturbParams pairs. */
    std::vector<std::pair<std::map<std::string, std::string>, perturbParams>> m_perturbRules;

    /* Vector that specifies sensor names. */
    std::vector<std::string> m_sensorNames;

    /* Generate a random number by selecting a correct random number generator */
    float getRandom(const randParams& rParams);

    /* Gaussian random number generator */
    float genGaussianRV(const randParams& params);

    /* Uniform random number generator */
    float genUniformRV(const randParams& params);

    /* Random generator */
    std::mt19937_64 m_generator;

    /* Flips a pose around the world y-z plane (flip left to right) */
    Augmenter::Pose flipPose(const Augmenter::Pose& pose);
};
