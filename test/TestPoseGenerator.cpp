/*******************************************************************************
*
* @file TestPoseGenerator.cpp
*
******************************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>

#include "gtest/gtest.h"
#include "poseGenerator.hpp"
#include <common/TestsDataPath.hpp>

namespace
{

// The fixture for testing class PoseGenerator
class PoseGeneratorTest : public ::testing::Test
{
protected:
    // Declare constructor.
    std::unique_ptr<PoseGenerator> testObject;

    // Declare example configRules and sensorNames.
    std::vector<std::pair<std::string, PoseGenerator::perturbParams>> configRules;
    std::vector<std::string> testSensorNames;

    // Give example perturbation parameters.
    PoseGenerator::perturbParams perturbParams1{
        .shift        = {"gaussian", 0.5, 0.34},
        .rotation     = {"gaussian", 4.0, 1.0},
        .forward      = {"gaussian", 0.8, 0.5},
        .sensor_yaw   = {"gaussian", 5.0, 3.0},
        .sensor_pitch = {"gaussian", 6.0, 3.0},
        .sensor_roll  = {"gaussian", 0, 0},
        .flip         = true,
    };
    PoseGenerator::perturbParams perturbParams2{
        .shift        = {"gaussian", 0.5, 0.34},
        .rotation     = {"uniform", 8.0, 1.0},
        .forward      = {"uniform", 0.8, 0.5},
        .sensor_yaw   = {"uniform", 5.0, 3.0},
        .sensor_pitch = {"gaussian", 6.0, 3.0},
        .sensor_roll  = {"gaussian", 2.0, 1.5},
        .flip         = false,
    };

    PoseGeneratorTest() {}
    virtual ~PoseGeneratorTest() {}
    virtual void SetUp()
    {
        // Give example string labels.
        std::string string1 = "road_type=highway user_label=stable";
        std::string string2 = "road_type=local user_label=stable";

        // Give example pairs and push to the vector "configRules".
        std::pair<std::string, PoseGenerator::perturbParams> pair1(string1, perturbParams1);
        std::pair<std::string, PoseGenerator::perturbParams> pair2(string2, perturbParams2);

        configRules.push_back(pair1);
        configRules.push_back(pair2);

        // Give example sensor names.
        testSensorNames = {"center", "pilot", "pilotPinhole"};
        testObject.reset(new PoseGenerator(configRules, testSensorNames, 1));
    }
    virtual void TearDown() {}
};

bool valueInBound(float64_t val, float64_t limit)
{
    return (std::abs(val) <= limit);
}

TEST_F(PoseGeneratorTest, TestGeneratePoses4vecFrame_L0)
{
    // Example csv file to retrieve video labels.
    std::string labelFileName = TestsDataPath::get() + "FILENAME.csv"; // YOUR FILE NAME

    projMetaData::projMetaTrace trace(labelFileName);
    std::vector<uint32_t> vecUseCounts(trace.getNumDatapoints(), 2);

    // For random number generation, we repeat several times to guarantee correct results
    const uint32_t trialNum = 100;
    for (uint32_t trial = 0; trial < trialNum; trial++)
    {
        std::vector<std::vector<Augmenter::Pose>> framePoses =
            testObject->generatePoses4vecFrames(vecUseCounts, labelFileName);

        // We expect that the length of framePoses (output) equlas to that of given trace (input).
        ASSERT_EQ(framePoses.size(), trace.getNumDatapoints());

        for (uint32_t i = 0; i < trace.getNumDatapoints(); ++i)
        {
            // We expect that the number of generated poses per frame equals to useCounts.
            ASSERT_EQ(framePoses[i].size(), static_cast<unsigned int>(vecUseCounts[i]));
        }

        for (uint32_t i = 0; i < trace.getNumDatapoints(); ++i)
        {
            for (uint32_t j = 0; j < vecUseCounts[i]; ++j)
            {
                PoseGenerator::perturbParams expectedParams;
                if (i < 2)
                {
                    expectedParams = perturbParams1;
                    if (j % 2)
                    {
                        // We expect every other pose to be flipped if flipping is enabled
                        ASSERT_TRUE(framePoses[i][j].flip);
                    }
                }
                else
                {
                    expectedParams = perturbParams2;
                }
                // We expect the generated random numbers to be bounded (-max, max).
                ASSERT_TRUE(valueInBound(framePoses[i][j].shift, expectedParams.shift.max));
                ASSERT_TRUE(valueInBound(framePoses[i][j].rotation, expectedParams.rotation.max));
                ASSERT_TRUE(valueInBound(framePoses[i][j].forward, expectedParams.forward.max));
                for (auto sensorName : testSensorNames)
                {
                    ASSERT_TRUE(valueInBound(framePoses[i][j].sensor_yaw[sensorName],
                                             expectedParams.sensor_yaw.max));
                    ASSERT_TRUE(valueInBound(framePoses[i][j].sensor_roll[sensorName],
                                             expectedParams.sensor_roll.max));
                    ASSERT_TRUE(valueInBound(framePoses[i][j].sensor_pitch[sensorName],
                                             expectedParams.sensor_pitch.max));
                }
            }
        }
    }
}

} // namespace
