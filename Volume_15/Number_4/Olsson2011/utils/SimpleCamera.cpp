/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#include "SimpleCamera.h"
#include <ctype.h>
#include <stdio.h>
#include "Win32ApiWrapper.h"

// helper function to check if shift button is pressed.
static bool isShiftPressed()
{
  return (( GetKeyState( VK_LSHIFT   ) < 0 ) || ( GetKeyState( VK_RSHIFT   ) < 0 ));
}


namespace chag
{


SimpleCamera::SimpleCamera()
{
  m_prevMousePosition = make_vector(-1, -1);
  m_transform = make_identity<float4x4>();
  m_base = make_identity<float3x3>();
  m_moveSpeed = 0.0f;
  m_strafeSpeed = 0.0f;
  m_rollSpeed = 0.0f;
  m_moveVel = 100.0f;
}



void SimpleCamera::init(const float3 &up, const float3 &fwd, const float3 &pos)
{
  m_base = make_matrix(cross(up, fwd), up, fwd);
  m_transform = make_matrix(m_base, pos);
}



const float3 SimpleCamera::getPosition() const
{
  return make_vector3(m_transform.c4);
}



const float3 SimpleCamera::getDirection() const
{
  return make_vector3(m_transform.c3);
}



const float3 SimpleCamera::getUp() const
{
  return make_vector3(m_transform.c2);
}



void SimpleCamera::pitch(float angleRadians)
{
  m_transform = m_transform * make_rotation_x<float4x4>(angleRadians);
}



void SimpleCamera::roll(float angleRadians)
{
  m_transform = m_transform * make_rotation_z<float4x4>(angleRadians);
}



void SimpleCamera::yaw(float angleRadians)
{
  // this will cause a truly local yaw, which causes tumbling, what we
  // probably want is a yaw around the global up axis, in local space.
  //
  // -> m_transform = m_transform * make_rotation_y<float4x4>(angleRadians);

  // Instead yaw around world up axis.
  //float3x3 m = make_rotation_y<float3x3>(angleRadians) * make_matrix3x3(m_transform);
  float3x3 m = make_rotation<float3x3>(m_base.c2, angleRadians) * make_matrix3x3(m_transform);
  
  m_transform = make_matrix(m, getPosition());
}



void SimpleCamera::move(float distance)
{
  m_transform = m_transform * make_translation(make_vector(0.0f, 0.0f, distance));
}



void SimpleCamera::strafe(float distance)
{
  m_transform = m_transform * make_translation(make_vector(distance, 0.0f, 0.0f));
}



const float4x4 SimpleCamera::getTransform() const 
{ 
  return m_transform; 
}



void SimpleCamera::setTransform(const float4x4 &tfm) 
{ 
  m_transform = tfm; 
}



bool SimpleCamera::handleKeyInput(uint8_t keyCode, bool down)
{
  const float vel = down ? m_moveVel : 0.0f;

	switch(tolower(keyCode))
	{
  case 'w':
    m_moveSpeed = vel;
    break;
  case 's':
    m_moveSpeed = -vel;
    break;
  case 'a':
    m_strafeSpeed = vel;
    break;
  case 'd':
    m_strafeSpeed = -vel;
    break;
  case 'q':
    m_rollSpeed = down ? 1.0f : 0.0f;
    break;
  case 'e':
    m_rollSpeed = down ? -1.0f : 0.0f;
    break;
  default:
    return false;
  };
  return true;
}



void SimpleCamera::handleMouseInput(const chag::int2 &mousePosition)
{
  if (m_prevMousePosition.x > 0 && m_prevMousePosition.y > 0)
  {
    chag::int2 delta = mousePosition - m_prevMousePosition;

    pitch(-float(delta.y) / 100.0f);
    yaw(-float(delta.x) / 100.0f);
  }
  m_prevMousePosition = mousePosition;
}



void SimpleCamera::resetMouseInputPosition()
{
  m_prevMousePosition = make_vector(-1, -1);
}



void SimpleCamera::update(float dt)
{
  const float speedBoost = isShiftPressed() ? 10.0f : 1.0f;
  move(speedBoost * m_moveSpeed * dt);
  strafe(speedBoost * m_strafeSpeed * dt);
  roll(speedBoost * m_rollSpeed * dt);
}


}; // namespace chag


