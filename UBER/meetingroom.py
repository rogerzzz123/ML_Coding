from sortedcontainers import SortedList
import unittest
import bisect
class MeetingScheduler:

    def __init__(self, rooms):
        self.booking={room: SortedList() for room in rooms}

    def book(self, startTime: int, endTime:int):
        if startTime>=endTime:
            return (False, "Invalid: Start time must be before end time")

        for room, meeting in self.booking.items():
            i = bisect.bisect_right(meeting, (startTime, float('inf')))
            if (i==0 or meeting[i-1][1]<=startTime) and (i==len(meeting) or meeting[i][0]>=endTime):
                meeting.add((startTime, endTime))
                return (True, room)
        return (False, "No available room")


class TestMeetingScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler=MeetingScheduler(["RoomA", "RoomB"]
        )
    
    def test_basecase(self):
        self.assertEqual(self.scheduler.book(10, 12)[0], (True))
        self.assertEqual(self.scheduler.book(10, 10), ((False, "Invalid: Start time must be before end time")))
    
    def test_muitiplecase1(self):
        self.scheduler.book(10, 12)
        self.assertEqual(self.scheduler.book(11, 13)[0], (True))
    
    def test_multiplecase2(self):
        self.scheduler.book(10,12)
        self.scheduler.book(10,12)
        self.assertEqual(self.scheduler.book(11, 13), ((False, "No available room")))
        

if __name__=="__main__":
    unittest.main()











    
# class TestMeetingScheduler(unittest.TestCase):

#     def setUp(self):
#         self.scheduler=MeetingScheduler(["RoomA", "RoomB"])
    
#     def test_base_case(self):
#         self.assertEqual(self.scheduler.book(10, 12)[0], True)
#         self.assertEqual(self.scheduler.book(10, 10), ((False, "Invalid: Start time must be before end time")))

#     def test_multiple_cases(self):
#         self.scheduler.book(10, 12)
#         self.assertEqual(self.scheduler.book(11,13)[0], True)

#     def test_all_room_filled(self):
#         self.scheduler.book(10, 12)
#         self.scheduler.book(10, 12)
#         self.assertEqual(self.scheduler.book(11,13)[0], False)


# if __name__ == "__main__":
#     unittest.main()