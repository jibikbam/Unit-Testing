/*******************************************************************************
 *
 * @file poseGenerator.cpp
 *
 ******************************************************************************/

#include <random> // for uniform_real_distribution() & normal_distribution()

#include "poseGenerator.hpp"

using std::string;
using std::map;
using std::vector;

PoseGenerator::PoseGenerator(const std::vector<std::pair<std::string, perturbParams>>& configRules,
                             std::vector<std::string> sensorNames, unsigned int seed)
    : m_sensorNames(sensorNames), m_generator(seed)
{
    for (auto rule : configRules)
    {
        // Splits a string ("key1=val1 key2=val2 ..") into a map of stringKey-val pairs.
        map<string, string> labelConditions = projMetaData::stringMapFromSplitString(rule.first);

        // Throw an error if the given labelCondition is not a stringLabel or invalid.
        for (auto condition : labelConditions)
        {
            if (projMetaData::isFieldNumeric(condition.first))
            {
                throw std::invalid_argument("poses cannot be generated based on numeric label:\"" +
                                            condition.first + "\"");
            }
            if (!projMetaData::isLabelValid(condition.first, condition.second))
            {
                throw std::invalid_argument("invalid label conditon: \"" + condition.first +
                                            "\":\"" + condition.second + "\"");
            }
        }
        // Make a pair<map=labelConditions, perturbParams=rule.second> and add to m_perturbRules.
        m_perturbRules.push_back(std::pair<map<string, string>, perturbParams>(labelConditions,
                                                                               rule.second));
    }
}

std::vector<std::vector<Augmenter::Pose>> PoseGenerator::generatePoses4vecFrames(
    std::vector<uint32_t> vecUseCounts,
    const std::string& labelsFileName)
{
    uint32_t numFrames = vecUseCounts.size();

    // Use projMetaTrace class to use its member functions: getNumDatapoints() & doLabelsMatch()
    projMetaData::projMetaTrace trace = projMetaData::projMetaTrace(labelsFileName);

    if (trace.getNumDatapoints() != numFrames)
    {
        throw std::invalid_argument("Trace has " + std::to_string(trace.getNumDatapoints()) +
                                    " frames, but use count has " + std::to_string(numFrames) + " entries.");
    }

    // Generate Poses for each frame
    std::vector<std::vector<Augmenter::Pose>> vecVecPoses = {};
    for (uint32_t i = 0; i < numFrames; ++i)
    {
        std::vector<Augmenter::Pose> Poses = generatePoses4oneFrame(vecUseCounts.at(i), i, trace);
        vecVecPoses.push_back(Poses);
    }

    if (vecVecPoses.size() != numFrames) {
        throw std::runtime_error("not all frames received poses - something is wrong");
    }
    return vecVecPoses;
}

std::vector<Augmenter::Pose> PoseGenerator::generateShuffledPoses(
        std::vector<uint32_t> vecUseCounts,
        const std::string& labelsFileName)
{
    std::vector<std::vector<Augmenter::Pose>> unshuffledPoses = generatePoses4vecFrames(vecUseCounts, labelsFileName);

    // Shuffle poses
    std::vector<Augmenter::Pose> flattenedPoses;
    for (const auto& vec : unshuffledPoses)
    {
        for (const auto& pose : vec)
        {
            flattenedPoses.push_back(pose);
        }
    }
    if (flattenedPoses.size() == 0)
    {
        return {};
    }
    // TODO: we should shuffle on disk instead of here (saves time re-reading/decoding h264)
    std::shuffle(flattenedPoses.begin(), flattenedPoses.end(), m_generator);
    // TODO: The augmenter crashes if the first pose is fipped - we should fix this
    while (flattenedPoses.at(0).flip)
    {
        std::shuffle(flattenedPoses.begin(), flattenedPoses.end(), m_generator);
    }
    return flattenedPoses;
}

std::vector<Augmenter::Pose> PoseGenerator::generatePoses4oneFrame(
    uint32_t useCount,
    uint32_t index,
    const projMetaData::projMetaTrace& trace)
{
    std::vector<Augmenter::Pose> vecPoses = {};
    if (useCount == 0)
    {
        return vecPoses;
    }

    // Find the first rule that applies to this frame among many rules.
    auto first_rule = std::find_if(m_perturbRules.begin(),
                                   m_perturbRules.end(),
                                   [&](std::pair<std::map<std::string, std::string>, perturbParams> rule) {
                                       return trace.doLabelsMatch(index, rule.first);
                                   });

    if (first_rule != m_perturbRules.end())
    {
        for (uint32_t i = 0; i < useCount; ++i)
        {
            Augmenter::Pose onePose = generateOnePose(first_rule[0].second);
            onePose.srcFrame = index;
            // Push flipped pose every other pose
            if (first_rule[0].second.flip && i % 2)
            {
                vecPoses.push_back(flipPose(onePose));
            }
            else
            {
                vecPoses.push_back(onePose);
            }
        }
    }

    if (vecPoses.size() == 0)
    {
        throw std::runtime_error("no perturbation rule found for frame " + index);
    }

    return vecPoses;
}

Augmenter::Pose PoseGenerator::generateOnePose(const perturbParams& params)
{
    Augmenter::Pose aPose = {};

    // Get random numbers for shift, rotation, and forward.
    aPose.shift    = getRandom(params.shift);
    aPose.rotation = getRandom(params.rotation);
    aPose.forward  = getRandom(params.forward);

    // Get random numbers for sensor_yaw, sensor_pitch, and sensor_roll for given sensors.
    for (auto sensorName : m_sensorNames)
    {
        float amountSensor_yaw = getRandom(params.sensor_yaw);
        aPose.sensor_yaw.insert(std::pair<std::string, float>(sensorName, amountSensor_yaw));
        float amountSensor_pitch = getRandom(params.sensor_pitch);
        aPose.sensor_pitch.insert(std::pair<std::string, float>(sensorName, amountSensor_pitch));
        float amountSensor_roll = getRandom(params.sensor_roll);
        aPose.sensor_roll.insert(std::pair<std::string, float>(sensorName, amountSensor_roll));
    }
    aPose.flip = false;

    return aPose;
}

Augmenter::Pose PoseGenerator::flipPose(const Augmenter::Pose& in)
{
    // When flipping a pose, signs of shift and rotation change.
    Augmenter::Pose pose = in;
    pose.flip            = true;
    pose.shift *= -1;
    pose.rotation *= -1;

    return pose;
}

float PoseGenerator::getRandom(const randParams& rParams)
{
    float num = 0;
    if ((rParams.distribution == "gaussian") || (rParams.distribution == "normal"))
    {
        num = genGaussianRV(rParams);
    }
    else if (rParams.distribution == "uniform")
    {
        num = genUniformRV(rParams);
    }
    else
    {
        throw std::invalid_argument("Unknown distribution type: " + rParams.distribution);
    }

    return num;
}

float PoseGenerator::genGaussianRV(const randParams& params)
{

    // Produce a random number according to a Gaussian distribution while making sure
    // that the number is bounded (-max, max).
    const double meanDist = 0;
    std::normal_distribution<double> distribution_norm(meanDist, params.stdDev);
    double numGauss = distribution_norm(m_generator);
    while ((numGauss < -params.max) || (numGauss > params.max))
    {
        numGauss = distribution_norm(m_generator);
    }

    return numGauss;
}

float PoseGenerator::genUniformRV(const randParams& params)
{
    // Produce a random number according to a uniform distribution.
    std::uniform_real_distribution<double> distribution_unif(-params.max, params.max);
    double numUnif = distribution_unif(m_generator);

    return numUnif;
}
