#include <donut/core/math/math.h>
using namespace donut;
using namespace donut::math;

#include "ResultsReadbackHandler.h"
#include "TrainingResults.h"

ResultsReadbackHandler::ResultsReadbackHandler(nvrhi::DeviceHandle device) : m_device(device)
{
    nvrhi::BufferDesc bufferDesc = {};
    bufferDesc.byteSize = sizeof(TrainingResults);
    bufferDesc.structStride = sizeof(TrainingResults);
    bufferDesc.canHaveUAVs = true;
    bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    bufferDesc.keepInitialState = true;
    bufferDesc.debugName = "trainingResultsBuffer";
    m_resultsBuffer = m_device->createBuffer(bufferDesc);

    bufferDesc.canHaveUAVs = false;
    bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
    bufferDesc.initialState = nvrhi::ResourceStates::Common;
    bufferDesc.debugName = "readbackResultsBuffer";
    for (int i = 0; i < c_bufferCount; i++)
    {
        m_readbackBuffers[i] = m_device->createBuffer(bufferDesc);
    }
}

void ResultsReadbackHandler::SyncResults(nvrhi::CommandListHandle commandList)
{
    const int writeIndex = m_currentIndex;
    const int readIndex = (m_currentIndex + 1) % c_bufferCount;
    m_hasResultsThisRun = false;

    if (m_hasResults)
    {
        void* pData = m_device->mapBuffer(m_readbackBuffers[readIndex], nvrhi::CpuAccessMode::Read);
        if (pData)
        {
            std::memcpy(&m_results, pData, sizeof(TrainingResults));
            m_hasResultsThisRun = true;
        }
        m_device->unmapBuffer(m_readbackBuffers[readIndex]);
    }

    // Copy the current  results
    commandList->copyBuffer(m_readbackBuffers[writeIndex], 0, m_resultsBuffer, 0, sizeof(TrainingResults));

    m_currentIndex = readIndex;

    // After one pass we have results
    if (!m_hasResults)
    {
        m_hasResults = true;
    }
}

nvrhi::BufferHandle ResultsReadbackHandler::GetResultsBuffers() const
{
    return m_resultsBuffer;
}


bool ResultsReadbackHandler::GetResults(TrainingResults& results) const
{
    if (m_hasResultsThisRun)
    {
        results = m_results;
        return true;
    }
    return false;
}

void ResultsReadbackHandler::Reset()
{
    m_currentIndex = 0;
    m_hasResults = false;
    m_hasResultsThisRun = false;
}
