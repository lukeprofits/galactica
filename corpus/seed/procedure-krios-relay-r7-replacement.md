---
title: Krios Relay R7 field replacement procedure
license: proprietary-sample
version: rev-4.2
collection: seed-procedures
doc_type: procedure
---

# Krios Relay R7 field replacement procedure

Applies to the Krios R7 series bus relay in Vantry HP-40 hydraulic control
cabinets. Read the whole procedure before starting; step 6 is time-critical.

## Required tools and parts

- Krios R7-B replacement relay, firmware 4.2 or later
- 4 mm hex driver and a torque wrench covering 6 to 25 N·m
- Krios service dongle, part number KD-118
- Anti-static wrist strap

## Safety isolation

1. Set the cabinet mode switch to LOCAL and confirm the amber MODE lamp is lit.
2. Open breaker B4, then breaker B1. Order matters: opening B1 first latches
   fault code E-217 and requires a full controller reset.
3. Verify zero volts at test points TP3 and TP4.
4. Apply the wrist strap to the cabinet ground stud, not to the door frame.

## Replacement

5. Release the two captive hex screws on the relay carrier and withdraw the
   carrier straight out. Do not rock it; the backplane pins bend at 3 degrees.
6. Within 90 seconds of removing the old relay, seat the replacement. The
   backplane supercapacitor holds configuration for roughly two minutes; beyond
   that the cabinet requires reprogramming with the KD-118 dongle.
7. Torque the carrier screws to 12 N·m in a diagonal pattern. Over-torque
   cracks the carrier ear, which is not a field-repairable part.

## Return to service

8. Close B1, then B4.
9. Hold the RESET button for 5 seconds. Expect a single long beep, then two
   short beeps, indicating firmware handshake success.
10. Confirm the relay reports state RUN and error register 0x00 on the panel.
11. Record the replacement in the cabinet log with the new relay serial number.

## Known failure signatures

- Two long beeps at step 9: firmware mismatch. The replacement is older than
  4.2 and must be updated with the dongle before use.
- Error E-217 after step 9: breakers were opened out of order. Perform a full
  controller reset, then repeat from step 8.
