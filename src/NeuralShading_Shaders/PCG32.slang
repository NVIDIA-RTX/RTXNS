/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#pragma once

struct PCG32
{
    uint64_t state, inc;

    // Initialize from initial state value and stream index
	__init(uint64_t init_state, uint64_t init_seq) 
    { 
		state = 0;
		inc = (init_seq << 1u) | 1u;
		nextUInt();
		state += init_state;
		nextUInt();
    }

    // Generates random 32 bit unsigned integer
    [mutating]
    uint nextUInt()
    {
		uint64_t oldstate = state;
		state = oldstate * 0x5851f42d4c957f2dULL + inc;
		uint xorshifted = (uint) (((oldstate >> 18u) ^ oldstate) >> 27u);
		uint rot = (uint) (oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

	// Generates random float
    [mutating]
	float nextFloat() 
    {
		uint x = (nextUInt() >> 9) | 0x3f800000u;
		return reinterpret<float>(x) - 1.0f;
	}
}
